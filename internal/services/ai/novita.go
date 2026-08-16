package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/shashank-sharma/backend/internal/config"
	"github.com/shashank-sharma/backend/internal/logger"
)

// NovitaClient implements the ChatAIClient interface using Novita's OpenAI-compatible API
type NovitaClient struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	streams    map[string]context.CancelFunc // Track active streams for cancellation
	streamsMu  sync.RWMutex
}

// NewNovitaClient creates a new Novita client with the provided API key
func NewNovitaClient(apiKey string) ChatAIClient {
	if apiKey == "" {
		logger.LogError("Novita API key is empty")
		return nil
	}

	client := &NovitaClient{
		apiKey:     apiKey,
		baseURL:    "https://api.novita.ai/openai/v1",
		httpClient: &http.Client{},
		streams:    make(map[string]context.CancelFunc),
	}

	return client
}

// Summarize implements the AIClient.Summarize method
func (c *NovitaClient) Summarize(ctx context.Context, req *SummarizeRequest) (*SummarizeResponse, error) {
	if req.Text == "" {
		return &SummarizeResponse{Summary: ""}, nil
	}

	maxLength := 150
	if req.MaxLength > 0 {
		maxLength = req.MaxLength
	}

	prompt := fmt.Sprintf(
		"Summarize the following text in a concise, informative way in %d characters or less:\n\n%s",
		maxLength,
		req.Text,
	)

	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Model:       config.DefaultNovitaModel,
		MaxTokens:   int(float64(maxLength) * 0.5),
		Temperature: 0.3,
		Stream:      false,
	}

	resp, err := c.SendChat(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary: %w", err)
	}

	return &SummarizeResponse{
		Summary: strings.TrimSpace(resp.Content),
	}, nil
}

// SuggestTags implements the AIClient.SuggestTags method
func (c *NovitaClient) SuggestTags(ctx context.Context, req *TagRequest) (*TagResponse, error) {
	maxTags := 5
	if req.MaxTags > 0 {
		maxTags = req.MaxTags
	}

	prompt := fmt.Sprintf(
		"Extract relevant tags from the following content. Return only %d tags as a comma-separated list:\n\nTitle: %s\n\nContent: %s",
		maxTags,
		req.Title,
		req.Content,
	)

	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Model:       config.DefaultNovitaModel,
		Temperature: 0.3,
		Stream:      false,
	}

	resp, err := c.SendChat(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to suggest tags: %w", err)
	}

	tagsStr := strings.TrimSpace(resp.Content)
	tags := strings.Split(tagsStr, ",")
	for i := range tags {
		tags[i] = strings.TrimSpace(tags[i])
	}

	return &TagResponse{
		Tags: tags,
	}, nil
}

// ClassifyContent implements the AIClient.ClassifyContent method
func (c *NovitaClient) ClassifyContent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error) {
	labelsStr := strings.Join(req.Labels, ", ")
	prompt := fmt.Sprintf(
		"Classify the following content into one of these categories: %s\n\nTitle: %s\n\nContent: %s\n\nReturn only the category name.",
		labelsStr,
		req.Title,
		req.Content,
	)

	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Model:       config.DefaultNovitaModel,
		Temperature: 0.3,
		Stream:      false,
	}

	resp, err := c.SendChat(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to classify content: %w", err)
	}

	label := strings.TrimSpace(resp.Content)

	return &ClassifyResponse{
		Label:      label,
		Confidence: 1.0, // Novita doesn't provide confidence scores
	}, nil
}

// RecommendContent implements the AIClient.RecommendContent method
func (c *NovitaClient) RecommendContent(ctx context.Context, req *RecommendRequest) (*RecommendResponse, error) {
	prompt := fmt.Sprintf(
		"Provide a relevance score (0-1) and explanation for how relevant this content is to the user.\n\nItem: %s\n\nReturn JSON: {\"score\": 0.5, \"explanation\": \"...\"}",
		req.Item.Title,
	)

	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Model:       config.DefaultNovitaModel,
		Temperature: 0.3,
		Stream:      false,
	}

	resp, err := c.SendChat(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to recommend content: %w", err)
	}

	var result struct {
		Score       float64 `json:"score"`
		Explanation string  `json:"explanation"`
	}

	if err := json.Unmarshal([]byte(resp.Content), &result); err != nil {
		return &RecommendResponse{
			Score:       0.5,
			Explanation: resp.Content,
		}, nil
	}

	return &RecommendResponse{
		Score:       result.Score,
		Explanation: result.Explanation,
	}, nil
}

// StreamChat streams a chat response chunk by chunk
func (c *NovitaClient) StreamChat(ctx context.Context, req *ChatRequest) (<-chan ChatChunk, error) {
	ch := make(chan ChatChunk, 100)

	streamID := req.StreamID
	if streamID == "" {
		streamID = fmt.Sprintf("stream_%d", ctx.Value("request_id"))
		if streamID == "" {
			streamID = fmt.Sprintf("stream_%d", ctx.Value("stream_id"))
		}
	}

	streamCtx, cancel := context.WithCancel(ctx)
	c.streamsMu.Lock()
	c.streams[streamID] = cancel
	c.streamsMu.Unlock()

	go func() {
		defer func() {
			c.streamsMu.Lock()
			delete(c.streams, streamID)
			c.streamsMu.Unlock()
			close(ch)
		}()

		err := c.streamChat(streamCtx, req, ch)
		if err != nil {
			ch <- ChatChunk{
				Done:  true,
				Error: err.Error(),
			}
		}
	}()

	return ch, nil
}

// streamChat performs the actual streaming request
func (c *NovitaClient) streamChat(ctx context.Context, req *ChatRequest, ch chan<- ChatChunk) error {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	messages := make([]map[string]interface{}, len(req.Messages))
	for i, msg := range req.Messages {
		messageMap := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle multimodal content (array) or simple text content (string)
		if contentArray, ok := msg.Content.([]interface{}); ok {
			// Multimodal content - array of content parts
			messageMap["content"] = contentArray
		} else {
			// Simple text content
			messageMap["content"] = msg.Content
		}

		messages[i] = messageMap
	}

	if req.SystemPrompt != "" {
		messages = append([]map[string]interface{}{
			{"role": "system", "content": req.SystemPrompt},
		}, messages...)
	}

	payload := map[string]interface{}{
		"model":    req.Model,
		"messages": messages,
		"stream":   true,
	}

	if req.Temperature > 0 {
		payload["temperature"] = req.Temperature
	}

	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return fmt.Errorf("API request failed: %d - %s", httpResp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(httpResp.Body)
	cumulativeTokens := 0
	cumulativeContent := ""

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "[DONE]" {
			ch <- ChatChunk{
				Content: cumulativeContent,
				Done:    true,
				Tokens:  cumulativeTokens,
			}
			break
		}

		var chunkData struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage struct {
				TotalTokens int `json:"total_tokens"`
			} `json:"usage"`
		}

		if err := json.Unmarshal([]byte(dataStr), &chunkData); err != nil {
			continue
		}

		if len(chunkData.Choices) > 0 {
			delta := chunkData.Choices[0].Delta.Content
			if delta != "" {
				cumulativeContent += delta
				ch <- ChatChunk{
					Content: delta,
					Done:    false,
				}
			}

			if chunkData.Choices[0].FinishReason != "" {
				// Best practice: Final chunk should not contain content, only signal completion
				// Content has already been sent incrementally via delta chunks
				ch <- ChatChunk{
					Content: "", // Empty - content already accumulated from deltas
					Done:    true,
					Tokens:  cumulativeTokens,
				}
				break
			}
		}

		if chunkData.Usage.TotalTokens > 0 {
			cumulativeTokens = chunkData.Usage.TotalTokens
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read stream: %w", err)
	}

	return nil
}

// SendChat sends a chat request and returns the complete response (non-streaming)
func (c *NovitaClient) SendChat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	messages := make([]map[string]interface{}, len(req.Messages))
	for i, msg := range req.Messages {
		messageMap := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle multimodal content (array) or simple text content (string)
		if contentArray, ok := msg.Content.([]interface{}); ok {
			// Multimodal content - array of content parts
			messageMap["content"] = contentArray
		} else {
			// Simple text content
			messageMap["content"] = msg.Content
		}

		messages[i] = messageMap
	}

	if req.SystemPrompt != "" {
		messages = append([]map[string]interface{}{
			{"role": "system", "content": req.SystemPrompt},
		}, messages...)
	}

	payload := map[string]interface{}{
		"model":    req.Model,
		"messages": messages,
		"stream":   false,
	}

	if req.Temperature > 0 {
		payload["temperature"] = req.Temperature
	}

	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed: %d - %s", httpResp.StatusCode, string(body))
	}

	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Model string `json:"model"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from API")
	}

	return &ChatResponse{
		Content: resp.Choices[0].Message.Content,
		Model:   resp.Model,
		Tokens:  resp.Usage.TotalTokens,
	}, nil
}

// CancelStream cancels an ongoing stream by stream ID
func (c *NovitaClient) CancelStream(ctx context.Context, streamID string) error {
	c.streamsMu.Lock()
	defer c.streamsMu.Unlock()

	cancel, exists := c.streams[streamID]
	if !exists {
		return fmt.Errorf("stream not found: %s", streamID)
	}

	cancel()
	delete(c.streams, streamID)
	return nil
}

// ListModels returns available models from Novita
func (c *NovitaClient) ListModels(ctx context.Context) ([]ModelInfo, error) {
	url := fmt.Sprintf("%s/models", c.baseURL)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed: %d - %s", httpResp.StatusCode, string(body))
	}

	var resp struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
			ContextSize int    `json:"context_size"`
			InputPrice  int64  `json:"input_token_price_per_m"`
			OutputPrice int64  `json:"output_token_price_per_m"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	models := make([]ModelInfo, len(resp.Data))
	for i, m := range resp.Data {
		name := m.DisplayName
		if name == "" {
			name = m.ID
		}

		models[i] = ModelInfo{
			ID:                m.ID,
			Name:              name,
			MaxTokens:         m.ContextSize,
			SupportsStreaming: true,
			SupportsTools:     false,
			Pricing: &ModelPricing{
				InputCostPerToken:  float64(m.InputPrice) / 1_000_000,
				OutputCostPerToken: float64(m.OutputPrice) / 1_000_000,
			},
		}

		provider := strings.Split(m.ID, "/")[0]
		models[i].Provider = provider
	}

	return models, nil
}
