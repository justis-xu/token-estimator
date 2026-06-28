package estimator

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type goldenEntry struct {
	Text   string `json:"text"`
	Model  string `json:"model"`
	Tokens int    `json:"tokens"`
}

func loadGolden(t *testing.T) []goldenEntry {
	t.Helper()
	// GOLDEN_DIR 单独指向 python/output；未设置时回退到 TOKEN_TABLES_DIR
	dir := os.Getenv("GOLDEN_DIR")
	if dir == "" {
		dir = os.Getenv("TOKEN_TABLES_DIR")
	}
	if dir == "" {
		t.Skip("TOKEN_TABLES_DIR not set")
	}
	f, err := os.Open(dir + "/golden.jsonl")
	if err != nil {
		t.Skipf("golden.jsonl not found: %v", err)
	}
	defer f.Close()

	var out []goldenEntry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var e goldenEntry
		if json.Unmarshal(sc.Bytes(), &e) == nil {
			out = append(out, e)
		}
	}
	return out
}

func TestLoadReturnsErrorWhenNoTablesExist(t *testing.T) {
	dir := t.TempDir()

	if _, err := Load(dir); err == nil {
		t.Fatal("Load() error = nil, want error for empty table directory")
	}
}

func TestLoadReturnsErrorWhenConfigIsMissing(t *testing.T) {
	dir := t.TempDir()
	table := make([]byte, cjkCount)
	if err := os.WriteFile(filepath.Join(dir, "doubao.bin"), table, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(dir); err == nil {
		t.Fatal("Load() error = nil, want error for missing config.json")
	}
}

func TestLoadReturnsErrorWhenTableLengthIsInvalid(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "doubao.bin"), []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := `{"default":{"zh":1.0,"mixed":1.0,"en":1.0}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(dir); err == nil {
		t.Fatal("Load() error = nil, want error for invalid table length")
	}
}

func TestLoadReturnsErrorWhenConfigJSONIsInvalid(t *testing.T) {
	dir := t.TempDir()
	table := make([]byte, cjkCount)
	if err := os.WriteFile(filepath.Join(dir, "doubao.bin"), table, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{bad json`), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(dir); err == nil {
		t.Fatal("Load() error = nil, want error for invalid config.json")
	}
}

func TestEstimateFallsBackToDefaultCJKWhenModelTableMissing(t *testing.T) {
	// 未知模型无词表，走除数兜底：2字 × (1/1.5) = 1.33 → round = 1
	tables := &Tables{
		bins:      map[string][]byte{},
		discounts: map[string]segmentedDiscount{"default": {Zh: 1.0, Mixed: 1.0, En: 1.0}},
	}

	got := tables.Estimate("你好", "qwen")
	if got != 1 {
		t.Fatalf("Estimate() = %d, want 1 from DefaultCJK divisor fallback", got)
	}
}

func TestEstimateDoesNotTreatASCIISpaceAsRareSymbol(t *testing.T) {
	tables := &Tables{
		bins:      map[string][]byte{},
		discounts: map[string]segmentedDiscount{"default": {Zh: 1.0, Mixed: 1.0, En: 1.0}},
	}

	got := tables.Estimate("Hello world", "unknown")
	if got != 4 {
		t.Fatalf("Estimate() = %d, want 4 for latin words with a low-cost space", got)
	}
}

func TestEstimateUsesModelSpecificWeightsBeforeDefaultWeights(t *testing.T) {
	flat1 := segmentedDiscount{Zh: 1.0, Mixed: 1.0, En: 1.0}
	tables := &Tables{
		bins:      map[string][]byte{},
		discounts: map[string]segmentedDiscount{"default": flat1, "gpt-4o": flat1},
		weights: map[string]heuristicWeights{
			"default": {ASCIISpace: 1.0},
			"gpt-4o":  {ASCIISpace: 0.2},
		},
	}

	if got := tables.Estimate("Hello world", "gpt-4o"); got != 4 {
		t.Fatalf("Estimate(gpt-4o) = %d, want model-specific low-cost space result 4", got)
	}
	if got := tables.Estimate("Hello world", "unknown-model"); got != 5 {
		t.Fatalf("Estimate(unknown-model) = %d, want default-weight result 5", got)
	}
}

func TestResolveKeyUsesSpecificGenerationBeforeFamilyFallback(t *testing.T) {
	cases := map[string]string{
		"Qwen2.5-72B-Instruct": "qwen2",
		"deepseek-v3":          "deepseek-v3",
		"DeepSeekV3":           "deepseek-v3",
		"glm-4-9b-chat":        "glm4",
		"GLM4":                 "glm4",
	}

	for model, want := range cases {
		if got := resolveKey(model); got != want {
			t.Fatalf("resolveKey(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestEstimateFallsBackToDefaultCJKTokenWhenDoubaoIsMissing(t *testing.T) {
	tables := &Tables{
		bins:      map[string][]byte{},
		discounts: map[string]segmentedDiscount{"default": {Zh: 1.0, Mixed: 1.0, En: 1.0}},
	}

	got := tables.Estimate("你好", "qwen")
	if got != 1 {
		t.Fatalf("Estimate() = %d, want 1 from DefaultCJK divisor fallback", got)
	}
}

func TestPackageEstimateReturnsErrorBeforeInit(t *testing.T) {
	old := defaultTables
	defaultTables = nil
	defer func() { defaultTables = old }()

	if _, err := Estimate("你好", "qwen"); err == nil {
		t.Fatal("Estimate() error = nil, want error before Init")
	}
}

func TestInitLoadsDefaultTables(t *testing.T) {
	dir := t.TempDir()
	table := make([]byte, cjkCount)
	for i := range table {
		table[i] = 1
	}
	if err := os.WriteFile(filepath.Join(dir, "doubao.bin"), table, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := `{"default":{"zh":1.0,"mixed":1.0,"en":1.0}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := Init(dir); err != nil {
		t.Fatalf("Init() error = %v, want nil", err)
	}

	got, err := Estimate("你好", "missing-model")
	if err != nil {
		t.Fatalf("Estimate() error = %v, want nil", err)
	}
	// missing-model 无词表，走除数兜底：2字 × (1/1.5) = 1.33 → round = 1
	if got != 1 {
		t.Fatalf("Estimate() = %d, want 1", got)
	}
}

func TestEstimateSmoke(t *testing.T) {
	dir := os.Getenv("TOKEN_TABLES_DIR")
	if dir == "" {
		t.Skip("TOKEN_TABLES_DIR not set")
	}
	tables, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	cases := []struct{ text, model string }{
		{"你好，世界！", "qwen"},
		{"Hello world", "gpt-4o"},
		{"こんにちは", "claude-opus-4-8"},
		{"deepseek模型分词测试", "deepseek-v3"},
		{"", "gpt-4"},
	}
	for _, c := range cases {
		n := tables.Estimate(c.text, c.model)
		if n < 0 {
			t.Errorf("Estimate(%q, %q) = %d, want >= 0", c.text, c.model, n)
		}
	}
}

func TestEstimateAccuracy(t *testing.T) {
	dir := os.Getenv("TOKEN_TABLES_DIR")
	if dir == "" {
		t.Skip("TOKEN_TABLES_DIR not set")
	}
	tables, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	golden := loadGolden(t)

	type stat struct {
		sumErr float64
		n      int
	}
	byModel := map[string]*stat{}

	for _, g := range golden {
		est := tables.Estimate(g.Text, g.Model)
		key := resolveKey(g.Model)
		if byModel[key] == nil {
			byModel[key] = &stat{}
		}
		absRel := math.Abs(float64(est-g.Tokens)) / math.Max(float64(g.Tokens), 1)
		byModel[key].sumErr += absRel
		byModel[key].n++
	}

	fmt.Printf("\n%-12s  %8s  %6s\n", "model", "samples", "MAE%")
	fmt.Println(strings.Repeat("-", 32))
	for key, s := range byModel {
		mae := s.sumErr / float64(s.n) * 100
		fmt.Printf("%-12s  %8d  %5.1f%%\n", key, s.n, mae)
		if mae > 15 {
			t.Errorf("model %s: MAE %.1f%% exceeds 15%% threshold", key, mae)
		}
	}
}
