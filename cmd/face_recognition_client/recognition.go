package main

import (
	"math"
	"sort"
)

type MatchResult struct {
	Name       string  `json:"name"`
	Similarity float32 `json:"similarity"`
	Accepted   bool    `json:"accepted"`
	Candidate  string  `json:"candidate,omitempty"`
}

// matchRules decide when the most similar person is accepted as the answer.
type matchRules struct {
	Threshold       float32 // lowest similarity accepted
	SingleThreshold float32 // the same, while only one person can be compared
	Margin          float32 // how far ahead of the second most similar person
}

// findBestMatch compares a face with the enrolled people. It returns nil when
// there is nobody to compare with.
func findBestMatch(embedding []float32, people []personEmbedding, rules matchRules) *MatchResult {
	matches := topMatches(embedding, people)
	if len(matches) == 0 {
		return nil
	}
	best := matches[0]
	accepted := false
	if len(matches) == 1 {
		// Nobody to tell this person apart from, so the bar is higher.
		accepted = best.Similarity >= rules.SingleThreshold
	} else {
		accepted = best.Similarity >= rules.Threshold && best.Similarity-matches[1].Similarity >= rules.Margin
	}
	if accepted {
		best.Accepted = true
		return &best
	}
	return &MatchResult{Name: "unknown", Similarity: best.Similarity, Candidate: best.Name}
}

// topMatches returns, for each person with profiles, the similarity of the
// profile closest to the face, most similar person first.
func topMatches(embedding []float32, people []personEmbedding) []MatchResult {
	if len(embedding) == 0 {
		return nil
	}
	results := make([]MatchResult, 0, len(people))
	for _, p := range people {
		if len(p.Profiles) == 0 {
			continue
		}
		best := float32(-1)
		for _, profile := range p.Profiles {
			if s := cosineSimilarity(embedding, profile.Embedding); s > best {
				best = s
			}
		}
		results = append(results, MatchResult{Name: p.Name, Similarity: best})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Similarity > results[j].Similarity })
	return results
}

func cosineSimilarity(a, b []float32) float32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		af, bf := float64(a[i]), float64(b[i])
		dot += af * bf
		na += af * af
		nb += bf * bf
	}
	den := math.Sqrt(na) * math.Sqrt(nb)
	if den == 0 {
		return 0
	}
	return float32(dot / den)
}

func normalizeL2(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	norm := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] * norm
	}
	return out
}
