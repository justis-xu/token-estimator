package estimator

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const (
	cjkStart = 0x4E00
	cjkEnd   = 0x9FFF
	cjkCount = cjkEnd - cjkStart + 1 // 20992

	// Built-in defaults used when no table / no discount is available.
	defaultDiscount = 0.9 // conservative; overridden by config.json["default"]
)

type heuristicWeights struct {
	DefaultCJK     float64 `json:"default_cjk"`
	LatinDivisor   float64 `json:"latin_divisor"`
	Hiragana       float64 `json:"hiragana"`
	Korean         float64 `json:"korean"`
	Digit          float64 `json:"digit"`
	Newline        float64 `json:"newline"`
	Tab            float64 `json:"tab"`
	ASCIISpace     float64 `json:"ascii_space"`
	CJKPunctuation float64 `json:"cjk_punctuation"`
	ASCIIPunct     float64 `json:"ascii_punct"`
	Other          float64 `json:"other"`
}

func defaultWeights() heuristicWeights {
	return heuristicWeights{
		DefaultCJK:     1.5,
		LatinDivisor:   4.0,
		Hiragana:       1.0,
		Korean:         1.5,
		Digit:          0.5,
		Newline:        0.5,
		Tab:            0.8,
		ASCIISpace:     0.2,
		CJKPunctuation: 1.0,
		ASCIIPunct:     0.7,
		Other:          3.0,
	}
}

// Tables holds per-model data loaded at startup.
type Tables struct {
	bins      map[string][]byte
	discounts map[string]float64
	weights   map[string]heuristicWeights
}

var defaultTables *Tables

// Init loads tables into the package-level estimator used by Estimate.
func Init(dir string) error {
	tables, err := Load(dir)
	if err != nil {
		return err
	}
	defaultTables = tables
	return nil
}

// Estimate uses tables previously loaded by Init.
func Estimate(text, model string) (int, error) {
	if defaultTables == nil {
		return 0, errors.New("estimator is not initialized")
	}
	return defaultTables.Estimate(text, model), nil
}

// Load reads all .bin files and config.json from dir into memory.
// dir defaults to the TOKEN_TABLES_DIR environment variable when empty.
func Load(dir string) (*Tables, error) {
	if dir == "" {
		dir = os.Getenv("TOKEN_TABLES_DIR")
	}
	if dir == "" {
		return nil, fmt.Errorf("TOKEN_TABLES_DIR is not set")
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir %s: %w", dir, err)
	}

	bins := make(map[string][]byte)
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".bin" {
			continue
		}
		key := strings.TrimSuffix(e.Name(), ".bin")
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", e.Name(), err)
		}
		// Validate length to avoid out-of-range panics at lookup time.
		if len(data) != cjkCount {
			return nil, fmt.Errorf("%s: expected %d bytes, got %d (corrupt or partial table)",
				e.Name(), cjkCount, len(data))
		}
		bins[key] = data
	}
	if len(bins) == 0 {
		return nil, fmt.Errorf("no token table .bin files found in %s", dir)
	}

	discounts := map[string]float64{}
	weights := map[string]heuristicWeights{"default": defaultWeights()}
	cfgData, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	cfg := map[string]json.RawMessage{}
	if err := json.Unmarshal(cfgData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}
	for key, raw := range cfg {
		if key == "weights" {
			parsedWeights, err := parseWeights(raw)
			if err != nil {
				return nil, fmt.Errorf("parse config.json weights: %w", err)
			}
			weights = parsedWeights
			continue
		}
		var discount float64
		if err := json.Unmarshal(raw, &discount); err != nil {
			return nil, fmt.Errorf("parse config.json discount %q: %w", key, err)
		}
		discounts[key] = discount
	}
	// Guarantee a default discount exists so estimation never multiplies by 0.
	if _, ok := discounts["default"]; !ok {
		discounts["default"] = defaultDiscount
	}

	return &Tables{bins: bins, discounts: discounts, weights: weights}, nil
}

func parseWeights(raw json.RawMessage) (map[string]heuristicWeights, error) {
	weightsByModel := map[string]heuristicWeights{}
	if err := json.Unmarshal(raw, &weightsByModel); err == nil {
		for key, weights := range weightsByModel {
			weightsByModel[key] = weights.withDefaults()
		}
		if _, ok := weightsByModel["default"]; !ok {
			weightsByModel["default"] = defaultWeights()
		}
		return weightsByModel, nil
	}

	var weights heuristicWeights
	if err := json.Unmarshal(raw, &weights); err != nil {
		return nil, err
	}
	return map[string]heuristicWeights{"default": weights.withDefaults()}, nil
}

func (w heuristicWeights) withDefaults() heuristicWeights {
	defaults := defaultWeights()
	if w.DefaultCJK == 0 {
		w.DefaultCJK = defaults.DefaultCJK
	}
	if w.LatinDivisor == 0 {
		w.LatinDivisor = defaults.LatinDivisor
	}
	if w.Hiragana == 0 {
		w.Hiragana = defaults.Hiragana
	}
	if w.Korean == 0 {
		w.Korean = defaults.Korean
	}
	if w.Digit == 0 {
		w.Digit = defaults.Digit
	}
	if w.Newline == 0 {
		w.Newline = defaults.Newline
	}
	if w.Tab == 0 {
		w.Tab = defaults.Tab
	}
	if w.ASCIISpace == 0 {
		w.ASCIISpace = defaults.ASCIISpace
	}
	if w.CJKPunctuation == 0 {
		w.CJKPunctuation = defaults.CJKPunctuation
	}
	if w.ASCIIPunct == 0 {
		w.ASCIIPunct = defaults.ASCIIPunct
	}
	if w.Other == 0 {
		w.Other = defaults.Other
	}
	return w
}

// resolveKey maps an arbitrary model string to an internal key.
func resolveKey(model string) string {
	m := strings.ToLower(model)

	// OpenAI reasoning series — must come before generic "gpt" check
	if strings.HasPrefix(m, "o1") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4") {
		return "gpt-4o"
	}
	// o200k_base family: gpt-4o, gpt-5, gpt-5.1, ...
	if strings.Contains(m, "gpt-4o") || strings.Contains(m, "gpt-5") {
		return "gpt-4o"
	}
	// cl100k_base family: gpt-4, gpt-4-turbo, gpt-3.5-turbo
	if strings.Contains(m, "gpt") {
		return "gpt-4"
	}
	if strings.Contains(m, "claude") {
		return "claude"
	}
	if strings.Contains(m, "qwen2") || strings.Contains(m, "qwen-2") {
		return "qwen2"
	}
	if strings.Contains(m, "qwen") {
		return "qwen"
	}
	if strings.Contains(m, "deepseek-v3") || strings.Contains(m, "deepseekv3") {
		return "deepseek-v3"
	}
	if strings.Contains(m, "deepseek") {
		return "deepseek"
	}
	if strings.Contains(m, "glm-4") || strings.Contains(m, "glm4") {
		return "glm4"
	}
	if strings.Contains(m, "glm") {
		return "glm"
	}
	if strings.Contains(m, "minimax") {
		return "minimax"
	}
	if strings.Contains(m, "kimi") || strings.Contains(m, "moonshot") {
		return "kimi"
	}
	if strings.Contains(m, "doubao") {
		return "doubao"
	}
	return "default"
}

// pickTable returns the best available table for key, or nil if none can be
// found. A nil table triggers the built-in default algorithm (defaultCJKToken
// per CJK char), so estimation still works for unknown/uncovered models.
func (t *Tables) pickTable(key string) []byte {
	if tbl, ok := t.bins[key]; ok {
		return tbl
	}
	if tbl, ok := t.bins["doubao"]; ok { // most conservative known table
		return tbl
	}
	return nil
}

func (t *Tables) discountFor(key string) float64 {
	if d, ok := t.discounts[key]; ok {
		return d
	}
	return t.discounts["default"] // always present (guaranteed by Load)
}

func (t *Tables) weightsFor(key string) heuristicWeights {
	if weights, ok := t.weights[key]; ok {
		return weights.withDefaults()
	}
	if weights, ok := t.weights["default"]; ok {
		return weights.withDefaults()
	}
	return defaultWeights()
}

// Estimate returns an approximate token count for text using the given model.
// The discount is applied globally to the whole heuristic estimate.
func (t *Tables) Estimate(text, model string) int {
	key := resolveKey(model)
	table := t.pickTable(key)
	discount := t.discountFor(key)
	weights := t.weightsFor(key)

	runes := []rune(text)
	var tokens float64

	i := 0
	for i < len(runes) {
		cp := runes[i]

		switch {
		// CJK Unified Ideographs (main block) — table lookup or default
		case cp >= cjkStart && cp <= cjkEnd:
			if table != nil {
				tokens += float64(table[cp-cjkStart])
			} else {
				tokens += weights.DefaultCJK
			}
			i++

		// CJK Extension A / Compatibility Ideographs (fallback)
		case (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0xF900 && cp <= 0xFAFF):
			tokens += weights.DefaultCJK
			i++

		// Latin letter run — scale with length
		case isLatin(cp):
			j := i + 1
			for j < len(runes) && isLatin(runes[j]) {
				j++
			}
			tokens += math.Ceil(float64(j-i) / weights.LatinDivisor)
			i = j

		// Hiragana / Katakana
		case (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF):
			tokens += weights.Hiragana
			i++

		// Korean syllables
		case cp >= 0xAC00 && cp <= 0xD7AF:
			tokens += weights.Korean
			i++

		// Digit run
		case unicode.IsDigit(cp):
			j := i + 1
			for j < len(runes) && unicode.IsDigit(runes[j]) {
				j++
			}
			tokens += float64(j-i) * weights.Digit
			i = j

		// Newlines
		case cp == '\n' || cp == '\r':
			tokens += weights.Newline
			i++

		// ASCII whitespace often merges into adjacent tokens, especially in
		// code, JSON, and Markdown indentation.
		case cp == '\t':
			tokens += weights.Tab
			i++
		case cp == ' ':
			tokens += weights.ASCIISpace
			i++

		// CJK / fullwidth / general punctuation (，。、；：？！ etc.)
		case (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFFEF):
			tokens += weights.CJKPunctuation
			i++

		// ASCII punctuation (printable, non-alphanumeric)
		case cp >= 0x21 && cp <= 0x7E && !unicode.IsLetter(cp) && !unicode.IsDigit(cp):
			tokens += weights.ASCIIPunct
			i++

		// Everything else (emoji, rare symbols, …)
		default:
			tokens += weights.Other
			i++
		}
	}

	out := int(tokens*discount + 0.5)
	if out == 0 && len(runes) > 0 {
		return 1
	}
	return out
}

func isLatin(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}
