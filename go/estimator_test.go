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
	dir := os.Getenv("TOKEN_TABLES_DIR")
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
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"default":1.0}`), 0o644); err != nil {
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

func TestEstimateFallsBackWhenModelTableIsMissing(t *testing.T) {
	doubao := make([]byte, cjkCount)
	for i := range doubao {
		doubao[i] = 2
	}
	tables := &Tables{
		bins:      map[string][]byte{"doubao": doubao},
		discounts: map[string]float64{"default": 1.0},
	}

	got := tables.Estimate("你好", "qwen")
	if got != 4 {
		t.Fatalf("Estimate() = %d, want 4 from doubao fallback table", got)
	}
}

func TestEstimateFallsBackToDefaultCJKTokenWhenDoubaoIsMissing(t *testing.T) {
	tables := &Tables{
		bins:      map[string][]byte{},
		discounts: map[string]float64{"default": 1.0},
	}

	got := tables.Estimate("你好", "qwen")
	if got != 3 {
		t.Fatalf("Estimate() = %d, want 3 from built-in CJK fallback", got)
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
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"default":1.0}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := Init(dir); err != nil {
		t.Fatalf("Init() error = %v, want nil", err)
	}

	got, err := Estimate("你好", "missing-model")
	if err != nil {
		t.Fatalf("Estimate() error = %v, want nil", err)
	}
	if got != 2 {
		t.Fatalf("Estimate() = %d, want 2", got)
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
