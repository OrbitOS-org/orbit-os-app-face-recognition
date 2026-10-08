package main

import (
	"os"
	"path/filepath"
	"testing"
)

func unit(values ...float32) []float32 { return normalizeL2(values) }

func TestMatchUsesTheClosestProfile(t *testing.T) {
	rules := matchRules{Threshold: 0.70, SingleThreshold: 0.92, Margin: 0.10}
	people := []personEmbedding{
		{Name: "Ana", Profiles: []faceProfile{{Embedding: unit(1, 0, 0)}, {Embedding: unit(0, 1, 0)}}}, // two lights
		{Name: "Rui", Profiles: []faceProfile{{Embedding: unit(0, 0, 1)}}},
		{Name: "Old", Profiles: nil}, // recorded by an earlier version: not compared
	}

	// Close to Ana's second recording, far from the first: still Ana.
	m := findBestMatch(unit(0.05, 1, 0.1), people, rules)
	if m == nil || !m.Accepted || m.Name != "Ana" || m.Similarity < 0.98 {
		t.Fatalf("match = %+v", m)
	}
	// Between Ana and Rui: nobody is far enough ahead.
	m = findBestMatch(unit(0, 1, 0.9), people, rules)
	if m == nil || m.Accepted || m.Name != "unknown" || m.Candidate != "Ana" {
		t.Fatalf("ambiguous face: %+v", m)
	}
	// One person to compare with: the higher threshold applies.
	alone := people[:1]
	if m = findBestMatch(unit(1, 0.5, 0), alone, rules); m == nil || m.Accepted {
		t.Fatalf("similarity 0.89 with one person: %+v", m)
	}
	if m = findBestMatch(unit(1, 0.2, 0), alone, rules); m == nil || !m.Accepted {
		t.Fatalf("similarity 0.98 with one person: %+v", m)
	}
	// Nobody with profiles: no answer at all.
	if m = findBestMatch(unit(1, 0, 0), people[2:], rules); m != nil {
		t.Fatalf("no profiles: %+v", m)
	}
}

func TestStoreKeepsRecordings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "face_db", "embeddings.json")
	s := NewEmbeddingStore(path)
	for i := 0; i < maxProfiles+2; i++ {
		if err := s.AddProfile("Ana Silva", unit(float32(i+1), 1, 0), 10+i); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.AddProfile("  ", unit(1), 1); err == nil {
		t.Fatal("empty name accepted")
	}

	again := NewEmbeddingStore(path)
	if err := again.Load(); err != nil {
		t.Fatal(err)
	}
	people := again.List()
	if len(people) != 1 || people[0].Name != "Ana Silva" || len(people[0].Profiles) != maxProfiles {
		t.Fatalf("after reload: %d people, %d profiles", len(people), len(people[0].Profiles))
	}
	if first := people[0].Profiles[0]; first.Samples != 12 { // the two oldest recordings were dropped
		t.Fatalf("oldest kept recording has %d samples, want 12", first.Samples)
	}
	if info, _ := os.Stat(path); info.Mode().Perm() != 0o600 {
		t.Fatalf("file mode %v, want 0600", info.Mode().Perm())
	}

	if removed, err := again.Remove("ana silva"); err != nil || !removed {
		t.Fatalf("remove: %v %v", removed, err)
	}
	if len(again.List()) != 0 {
		t.Fatal("still listed after remove")
	}
}

// A file written before profiles existed: the names are kept, the face data is not used.
func TestStoreReadsEarlierFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "embeddings.json")
	old := `{"people":[{"name":"Ana","embedding":[0.6,0.8],"samples":3,"updated_at":"2026-10-01T10:00:00Z"}]}`
	if err := os.WriteFile(path, []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}
	s := NewEmbeddingStore(path)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	people := s.List()
	if len(people) != 1 || people[0].Name != "Ana" || len(people[0].Profiles) != 0 {
		t.Fatalf("earlier file: %+v", people)
	}
	if m := findBestMatch(unit(0.6, 0.8), people, matchRules{Threshold: 0.5, SingleThreshold: 0.5}); m != nil {
		t.Fatalf("earlier data was compared: %+v", m)
	}
	// Recording again makes the person usable.
	if err := s.AddProfile("Ana", unit(0.6, 0.8), 5); err != nil {
		t.Fatal(err)
	}
	if len(s.List()[0].Profiles) != 1 {
		t.Fatal("no profile after recording again")
	}
}

func TestMatchRulesFollowTheThreshold(t *testing.T) {
	c := &FaceClient{cfg: hardcodedConfig()}
	c.store = NewEmbeddingStore(filepath.Join(t.TempDir(), "face_db", "embeddings.json"))
	c.loadSettings()
	if r := c.matchRules(); r.Threshold != 0.40 || !near(float64(r.SingleThreshold), 0.45, 1e-6) || r.Margin != 0.08 {
		t.Fatalf("defaults: %+v", r)
	}
	if err := c.setMatchThreshold(0.50); err != nil {
		t.Fatal(err)
	}
	if r := c.matchRules(); r.Threshold != 0.50 || !near(float64(r.SingleThreshold), 0.55, 1e-6) {
		t.Fatalf("after change: %+v", r)
	}
	if err := c.setMatchThreshold(0.1); err == nil {
		t.Fatal("0.1 accepted")
	}
	// The choice is still there after a restart.
	again := &FaceClient{cfg: hardcodedConfig(), store: c.store}
	again.loadSettings()
	if again.matchThreshold != 0.50 {
		t.Fatalf("after restart: %v", again.matchThreshold)
	}
	// A choice made for another model is not used.
	if err := os.WriteFile(c.settingsPath(), []byte(`{"match_threshold":0.7,"profile_version":2}`), 0o600); err != nil {
		t.Fatal(err)
	}
	again.loadSettings()
	if again.matchThreshold != 0.40 {
		t.Fatalf("choice for another model: %v", again.matchThreshold)
	}
}
