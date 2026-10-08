package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/logger"
)

// The match threshold can be changed from the page; the choice is kept in a
// file next to the enrolled people.

const (
	matchThresholdMin  = 0.25
	matchThresholdMax  = 0.80
	singleThresholdMax = 0.90
)

type savedSettings struct {
	MatchThreshold float32 `json:"match_threshold"`
	// The threshold only means something for the model it was chosen with.
	ProfileVersion int `json:"profile_version"`
}

func (c *FaceClient) settingsPath() string {
	return filepath.Join(filepath.Dir(c.store.path), "settings.json")
}

// loadSettings reads the saved choice, if there is one. Call it after the store is loaded.
func (c *FaceClient) loadSettings() {
	c.matchThreshold = c.cfg.MatchThreshold
	data, err := os.ReadFile(c.settingsPath())
	if err != nil {
		return
	}
	var saved savedSettings
	if err := json.Unmarshal(data, &saved); err != nil {
		logger.Warnf(faceTag, "settings file ignored: %v", err)
		return
	}
	if saved.ProfileVersion == profileVersion && saved.MatchThreshold >= matchThresholdMin && saved.MatchThreshold <= matchThresholdMax {
		c.matchThreshold = saved.MatchThreshold
	}
}

// setMatchThreshold changes the threshold and saves the choice.
func (c *FaceClient) setMatchThreshold(v float32) error {
	if v < matchThresholdMin || v > matchThresholdMax {
		return fmt.Errorf("the match threshold must be between %.2f and %.2f", matchThresholdMin, matchThresholdMax)
	}
	c.stateMu.Lock()
	c.matchThreshold = v
	c.stateMu.Unlock()

	data, err := json.MarshalIndent(savedSettings{MatchThreshold: v, ProfileVersion: profileVersion}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.settingsPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(c.settingsPath(), data, 0o600)
}

// matchRules returns the rules in force. While only one person can be
// compared there is nobody to tell that person apart from, so the bar is
// higher by the same distance as in the defaults.
func (c *FaceClient) matchRules() matchRules {
	c.stateMu.RLock()
	threshold := c.matchThreshold
	c.stateMu.RUnlock()

	extra := c.cfg.SinglePersonThreshold - c.cfg.MatchThreshold
	if extra < 0 {
		extra = 0
	}
	return matchRules{
		Threshold:       threshold,
		SingleThreshold: min(threshold+extra, singleThresholdMax),
		Margin:          c.cfg.SecondBestMargin,
	}
}
