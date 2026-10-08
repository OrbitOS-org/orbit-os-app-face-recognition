package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// profileVersion says which model took the embeddings and how the face
	// was prepared for it. Embeddings from another model, or of a face
	// prepared in another way, cannot be compared with new faces, so profiles
	// of another version are not used: the person stays in the list, marked
	// to be recorded again.
	profileVersion = 3 // SFace, four-point alignment

	// A person keeps the last recordings as separate profiles (daylight,
	// lamp light, with glasses...). A face is compared with each of them.
	maxProfiles = 5
)

// faceProfile is one recording of a person: the average of the embeddings taken during it.
type faceProfile struct {
	Embedding  []float32 `json:"embedding"`
	Samples    int       `json:"samples,omitempty"`
	RecordedAt string    `json:"recorded_at,omitempty"`
}

type personEmbedding struct {
	Name      string        `json:"name"`
	Profiles  []faceProfile `json:"profiles,omitempty"`
	UpdatedAt string        `json:"updated_at,omitempty"`
}

type embeddingsFile struct {
	Version int               `json:"version"`
	People  []personEmbedding `json:"people"`
}

// EmbeddingStore keeps the enrolled people, in memory and in a JSON file.
type EmbeddingStore struct {
	path   string
	mu     sync.RWMutex
	people map[string]personEmbedding
}

func NewEmbeddingStore(path string) *EmbeddingStore {
	return &EmbeddingStore{
		path:   path,
		people: make(map[string]personEmbedding),
	}
}

func (s *EmbeddingStore) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if resolved := resolveEmbeddingsPath(s.path); resolved != "" {
		s.path = resolved
	}

	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read embeddings: %w", err)
	}

	var payload embeddingsFile
	if err := json.Unmarshal(data, &payload); err != nil {
		return fmt.Errorf("unmarshal embeddings: %w", err)
	}

	s.people = make(map[string]personEmbedding, len(payload.People))
	for _, p := range payload.People {
		key := normalizePersonName(p.Name)
		if key == "" {
			continue
		}
		if payload.Version != profileVersion {
			p.Profiles = nil // recorded by an earlier version: kept by name, to be recorded again
		}
		s.people[key] = p
	}
	return nil
}

// save writes the file. Call it with mu held.
func (s *EmbeddingStore) save() error {
	payload := embeddingsFile{Version: profileVersion, People: make([]personEmbedding, 0, len(s.people))}
	for _, p := range s.people {
		payload.People = append(payload.People, p)
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal embeddings: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("mkdir embeddings dir: %w", err)
	}
	// Names and face data: readable by the app only.
	if err := os.WriteFile(s.path, data, 0o600); err != nil {
		return fmt.Errorf("write embeddings: %w", err)
	}
	return nil
}

// AddProfile adds one recording to a person, creating the person when new.
// Only the last maxProfiles recordings are kept.
func (s *EmbeddingStore) AddProfile(name string, emb []float32, samples int) error {
	key := normalizePersonName(name)
	if key == "" {
		return fmt.Errorf("name is required")
	}
	if len(emb) == 0 {
		return fmt.Errorf("embedding is empty")
	}

	now := time.Now().UTC().Format(time.RFC3339)
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.people[key]
	if !ok {
		entry = personEmbedding{Name: strings.TrimSpace(name)}
	}
	entry.Profiles = append(entry.Profiles, faceProfile{
		Embedding:  append([]float32(nil), emb...),
		Samples:    samples,
		RecordedAt: now,
	})
	if extra := len(entry.Profiles) - maxProfiles; extra > 0 {
		entry.Profiles = append([]faceProfile(nil), entry.Profiles[extra:]...)
	}
	entry.UpdatedAt = now
	s.people[key] = entry
	return s.save()
}

func (s *EmbeddingStore) Remove(name string) (bool, error) {
	key := normalizePersonName(name)
	if key == "" {
		return false, fmt.Errorf("name is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.people[key]; !ok {
		return false, nil
	}
	delete(s.people, key)
	return true, s.save()
}

// List returns the people. The result shares the embeddings with the store:
// they are never changed once recorded, so read them, do not write.
func (s *EmbeddingStore) List() []personEmbedding {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]personEmbedding, 0, len(s.people))
	for _, p := range s.people {
		p.Profiles = append([]faceProfile(nil), p.Profiles...)
		out = append(out, p)
	}
	return out
}

func normalizePersonName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func resolveEmbeddingsPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	// Keep Go app data isolated: resolve only local app-relative paths.
	candidates := []string{path}
	if !filepath.IsAbs(path) {
		candidates = append(candidates, filepath.Join("..", path), filepath.Join("..", "..", path))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return path
}
