package estimator

import (
	"encoding/binary"
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
	cjkCount = cjkEnd - cjkStart + 1 // 20992 个汉字

	// 文本类别阈值（CJK 字符在 rune 切片中的占比）
	zhThreshold = 0.6 // ≥ zhThreshold → "zh"
	enThreshold = 0.1 // ≤ enThreshold → "en"；否则 "mixed"

	// 无词表 / 无 discount 时的兜底值
	defaultDiscount = 0.9 // 保守值；被 config.json["default"] 覆盖
)

// segmentedDiscount 存储各文本类别（zh/mixed/en）的独立 discount 系数。
type segmentedDiscount struct {
	Zh    float64 `json:"zh"`
	Mixed float64 `json:"mixed"`
	En    float64 `json:"en"`
}

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
		DefaultCJK:     1.5, // 无词表时 CJK 兜底除数：1/1.5≈0.667 token/字（withDefaults 保证非零，避免除零）
		LatinDivisor:   4.0, // 拉丁字母：ceil(词长 / 4) ≈ BPE 平均合并率
		Hiragana:       1.0, // 平假名 / 片假名：粒度接近汉字
		Korean:         1.5, // 韩文音节：切分较细，保守高估
		Digit:          0.5, // 数字：多位数通常合并，平均 2 位 ≈ 1 token
		Newline:        0.5, // 换行符：各模型行为一致，约 0.5 token
		Tab:            0.8, // Tab：缩进场景下多数独立成 token
		ASCIISpace:     0.2, // ASCII 空格：常与相邻 token 合并，权重极低
		CJKPunctuation: 1.0, // 中文标点 / 全角符号：通常独立成 token
		ASCIIPunct:     0.7, // ASCII 标点：多数独立或两两合并
		Other:          3.0, // 其余（emoji、罕见符号）：编码复杂，保守高估
	}
}

// Tables 持有启动时一次性加载的所有模型数据。
type Tables struct {
	bins      map[string][]byte
	bigrams   map[string]map[uint32]byte // 模型 key → 高频词表
	discounts map[string]segmentedDiscount
	weights   map[string]heuristicWeights
}

var defaultTables *Tables

// Init 将词表加载到包级估算器，供 Estimate 函数使用。
func Init(dir string) error {
	tables, err := Load(dir)
	if err != nil {
		return err
	}
	defaultTables = tables
	return nil
}

// Estimate 使用 Init 预加载的词表估算 token 数。
func Estimate(text, model string) (int, error) {
	if defaultTables == nil {
		return 0, errors.New("estimator 未初始化，请先调用 Init")
	}
	return defaultTables.Estimate(text, model), nil
}

// Load 将 dir 目录下的所有 .bin / .bigram 文件和 config.json 读入内存。
// dir 为空时读取环境变量 TOKEN_TABLES_DIR。
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
	bigrams := make(map[string]map[uint32]byte)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		switch filepath.Ext(name) {
		case ".bin":
			key := strings.TrimSuffix(name, ".bin")
			data, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", name, err)
			}
			if len(data) != cjkCount {
				return nil, fmt.Errorf("%s: expected %d bytes, got %d (corrupt or partial table)",
					name, cjkCount, len(data))
			}
			bins[key] = data
		case ".bigram":
			key := strings.TrimSuffix(name, ".bigram")
			data, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", name, err)
			}
			bg, err := parseBigramBin(data)
			if err != nil {
				return nil, fmt.Errorf("parse %s: %w", name, err)
			}
			bigrams[key] = bg
		}
	}
	if len(bins) == 0 {
		return nil, fmt.Errorf("no token table .bin files found in %s", dir)
	}

	discounts := map[string]segmentedDiscount{}
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
		// 优先解析分段 discount；兼容旧格式的单一浮点数
		var sd segmentedDiscount
		if err := json.Unmarshal(raw, &sd); err == nil && (sd.Zh > 0 || sd.Mixed > 0 || sd.En > 0) {
			discounts[key] = sd
			continue
		}
		var flat float64
		if err := json.Unmarshal(raw, &flat); err != nil {
			return nil, fmt.Errorf("parse config.json discount %q: %w", key, err)
		}
		discounts[key] = segmentedDiscount{Zh: flat, Mixed: flat, En: flat}
	}
	// 保证 default discount 始终存在，避免估算结果乘以 0
	if _, ok := discounts["default"]; !ok {
		discounts["default"] = segmentedDiscount{Zh: defaultDiscount, Mixed: defaultDiscount, En: defaultDiscount}
	}

	return &Tables{bins: bins, bigrams: bigrams, discounts: discounts, weights: weights}, nil
}

func parseBigramBin(data []byte) (map[uint32]byte, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("bigram: data too short")
	}
	n := binary.BigEndian.Uint32(data[:4])
	if len(data) != 4+int(n)*5 {
		return nil, fmt.Errorf("bigram: invalid length: expected %d bytes, got %d", 4+int(n)*5, len(data))
	}
	m := make(map[uint32]byte, n)
	for i := range int(n) {
		off := 4 + i*5
		off1 := uint32(binary.BigEndian.Uint16(data[off:]))
		off2 := uint32(binary.BigEndian.Uint16(data[off+2:]))
		m[(off1<<16)|off2] = data[off+4]
	}
	return m, nil
}

func (t *Tables) bigramFor(key string) map[uint32]byte {
	return t.bigrams[key] // 无词表时返回 nil，调用方跳过高频词查找
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

// resolveKey 将任意模型名映射到内部 key。
func resolveKey(model string) string {
	m := strings.ToLower(model)

	// OpenAI 推理系列：必须在通用 "gpt" 匹配之前处理
	if strings.HasPrefix(m, "o1") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4") {
		return "gpt-4o"
	}
	// o200k_base 系列：gpt-4o, gpt-5, gpt-5.1, ...
	if strings.Contains(m, "gpt-4o") || strings.Contains(m, "gpt-5") {
		return "gpt-4o"
	}
	// cl100k_base 系列：gpt-4, gpt-4-turbo, gpt-3.5-turbo
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

// pickTable 返回 key 对应的词表；找不到则返回 nil，
// 由 DefaultCJK 权重（经 calculate_weights.py 校准）兜底。
func (t *Tables) pickTable(key string) []byte {
	return t.bins[key]
}

// classifyText 按 CJK 字符占比返回 "zh"、"en" 或 "mixed"。
func classifyText(runes []rune) string {
	if len(runes) == 0 {
		return "mixed"
	}
	cjk := 0
	for _, r := range runes {
		if r >= cjkStart && r <= cjkEnd {
			cjk++
		}
	}
	ratio := float64(cjk) / float64(len(runes))
	if ratio >= zhThreshold {
		return "zh"
	}
	if ratio <= enThreshold {
		return "en"
	}
	return "mixed"
}

func (t *Tables) discountFor(key, textClass string) float64 {
	sd, ok := t.discounts[key]
	if !ok {
		sd = t.discounts["default"]
	}
	switch textClass {
	case "zh":
		return sd.Zh
	case "en":
		return sd.En
	default:
		return sd.Mixed
	}
}

func (t *Tables) weightsFor(key string) heuristicWeights {
	if weights, ok := t.weights[key]; ok {
		return weights.withDefaults()
	}
	// 未知模型走硬编码默认权重，不依赖 config.json 的 "default" 段
	return defaultWeights()
}

// Estimate 返回文本的估算 token 数，discount 根据 CJK 字符占比（zh/mixed/en）选取。
func (t *Tables) Estimate(text, model string) int {
	key := resolveKey(model)
	table := t.pickTable(key)
	weights := t.weightsFor(key)

	runes := []rune(text)
	textClass := classifyText(runes)
	discount := t.discountFor(key, textClass)
	bg := t.bigramFor(key)
	// bigramTokens：高频词表精确命中，不参与 discount 缩放
	// heuristicTokens：其余启发式估算，最终乘以 discount
	var bigramTokens, heuristicTokens float64

	i := 0
	for i < len(runes) {
		cp := runes[i]

		switch {
		// CJK 基本区：优先查高频词表 → 单字表 → 兜底系数
		case cp >= cjkStart && cp <= cjkEnd:
			if bg != nil && i+1 < len(runes) {
				cp2 := runes[i+1]
				if cp2 >= cjkStart && cp2 <= cjkEnd {
					bgKey := uint32(cp-cjkStart)<<16 | uint32(cp2-cjkStart)
					if count, ok := bg[bgKey]; ok {
						bigramTokens += float64(count)
						i += 2
						continue
					}
				}
			}
			if table != nil {
				heuristicTokens += float64(table[cp-cjkStart])
			} else {
				heuristicTokens += 1.0 / weights.DefaultCJK
			}
			i++

		// CJK 扩展区 / 兼容汉字：UTF-8 三字节，BPE 按字节切分 → 固定 3 token/字
		case (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0xF900 && cp <= 0xFAFF):
			heuristicTokens += 3.0
			i++

		// 拉丁字母连续段：按词长分桶
		case isLatin(cp):
			j := i + 1
			for j < len(runes) && isLatin(runes[j]) {
				j++
			}
			heuristicTokens += math.Ceil(float64(j-i) / weights.LatinDivisor)
			i = j

		// 平假名 / 片假名
		case (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF):
			heuristicTokens += weights.Hiragana
			i++

		// 韩文音节
		case cp >= 0xAC00 && cp <= 0xD7AF:
			heuristicTokens += weights.Korean
			i++

		// 数字连续段
		case unicode.IsDigit(cp):
			j := i + 1
			for j < len(runes) && unicode.IsDigit(runes[j]) {
				j++
			}
			heuristicTokens += float64(j-i) * weights.Digit
			i = j

		// 换行符
		case cp == '\n' || cp == '\r':
			heuristicTokens += weights.Newline
			i++

		// Tab（代码/JSON/Markdown 缩进场景下常见）
		case cp == '\t':
			heuristicTokens += weights.Tab
			i++

		// ASCII 空格
		case cp == ' ':
			heuristicTokens += weights.ASCIISpace
			i++

		// 中文标点 / 全角符号（，。、；：？！等）
		case (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFFEF):
			heuristicTokens += weights.CJKPunctuation
			i++

		// ASCII 标点（可打印非字母数字）
		case cp >= 0x21 && cp <= 0x7E && !unicode.IsLetter(cp) && !unicode.IsDigit(cp):
			heuristicTokens += weights.ASCIIPunct
			i++

		// 其余：emoji、罕见符号等
		default:
			heuristicTokens += weights.Other
			i++
		}
	}

	out := int(bigramTokens+heuristicTokens*discount + 0.5)
	if out == 0 && len(runes) > 0 {
		return 1
	}
	return out
}

func isLatin(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}
