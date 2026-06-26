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
	defaultCJKToken = 1.5 // per-CJK-char estimate when no table is loaded
)

// Tables holds per-model data loaded at startup.
type Tables struct {
	bins      map[string][]byte
	discounts map[string]float64
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
	cfgData, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	if err := json.Unmarshal(cfgData, &discounts); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}
	// Guarantee a default discount exists so estimation never multiplies by 0.
	if _, ok := discounts["default"]; !ok {
		discounts["default"] = defaultDiscount
	}

	return &Tables{bins: bins, discounts: discounts}, nil
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
	if strings.Contains(m, "qwen") {
		return "qwen"
	}
	if strings.Contains(m, "deepseek") {
		return "deepseek"
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

// Estimate returns an approximate token count for text using the given model.
// The discount is applied globally to the whole heuristic estimate.
func (t *Tables) Estimate(text, model string) int {
	key := resolveKey(model)
	table := t.pickTable(key)
	discount := t.discountFor(key)

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
				tokens += defaultCJKToken
			}
			i++

		// CJK Extension A / Compatibility Ideographs (fallback)
		case (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0xF900 && cp <= 0xFAFF):
			tokens += defaultCJKToken
			i++

		// Latin letter run — scale with length
		case isLatin(cp):
			j := i + 1
			for j < len(runes) && isLatin(runes[j]) {
				j++
			}
			tokens += math.Ceil(float64(j-i) / 4.0)
			i = j

		// Hiragana / Katakana
		case (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF):
			tokens += 1.0
			i++

		// Korean syllables
		case cp >= 0xAC00 && cp <= 0xD7AF:
			tokens += 1.5
			i++

		// Digit run
		case unicode.IsDigit(cp):
			j := i + 1
			for j < len(runes) && unicode.IsDigit(runes[j]) {
				j++
			}
			tokens += float64(j-i) * 0.5
			i = j

		// Newlines
		case cp == '\n' || cp == '\r':
			tokens += 1.0
			i++

		// CJK / fullwidth / general punctuation (，。、；：？！ etc.)
		case (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFFEF):
			tokens += 1.0
			i++

		// ASCII punctuation (printable, non-alphanumeric)
		case cp >= 0x21 && cp <= 0x7E && !unicode.IsLetter(cp) && !unicode.IsDigit(cp):
			tokens += 0.7
			i++

		// Everything else (emoji, rare symbols, …)
		default:
			tokens += 3.0
			i++
		}
	}

	return int(tokens*discount + 0.5)
}

func isLatin(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}
