package main

import (
	"regexp"
	"strings"
)

// --- TEXT NORMALIZATION ---

func normalizeText(text string) string {
	ConfigMu.RLock()
	defer ConfigMu.RUnlock()

	// 1. High-Performance Plain Text Replacements (O(N) pass)
	if PlainReplacer != nil {
		text = PlainReplacer.Replace(text)
	}

	// 2. Regex Replacements (Only for complex patterns)
	lowerText := strings.ToLower(text)
	for _, rule := range RegexRules {
		if rule.Guard != "" && !strings.Contains(lowerText, rule.Guard) {
			continue
		}
		text = rule.Re.ReplaceAllString(text, rule.Replacement)
	}

	// 3. Advanced Phrasing & Prosodic Emphasis
	text = applyProsodicEmphasis(text)
	return heuristicPhrasing(text)
}

func applyProsodicEmphasis(text string) string {
	// Detects: word followed by multiple ! or ? (e.g., "Wait!!!")
	// 1. Injects a comma-pause BEFORE for "gravity"
	// 2. Uppercases the word for Piper's "energy" boost
	// 3. Normalizes punctuation to a consistent emphatic ending
	reShout := regexp.MustCompile(`(\b[a-zA-Z]{2,})([!]{2,}|[?]{2,})`)
	return reShout.ReplaceAllStringFunc(text, func(m string) string {
		parts := reShout.FindStringSubmatch(m)
		if len(parts) < 3 {
			return m
		}
		// A preceding comma forces a slight pause/intonation reset which makes the shout impactful
		return ", " + strings.ToUpper(parts[1]) + parts[2][:1] + "!"
	})
}

func heuristicPhrasing(text string) string {
	// A. Convert newlines to periods if no punctuation exists
	// This handles "list-like" bad writing where people skip dots at line ends
	lines := strings.Split(text, "\n")
	changed := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 0 {
			lastChar := trimmed[len(trimmed)-1]
			if lastChar != '.' && lastChar != '!' && lastChar != '?' && lastChar != ',' && lastChar != ':' && lastChar != ';' {
				lines[i] = trimmed + "."
				changed = true
			}
		}
	}
	if changed {
		text = strings.Join(lines, " ")
	}

	// B. Force a "breath" every 18 words to prevent run-on sentences
	words := strings.Fields(text)
	if len(words) > 20 {
		var b strings.Builder
		wordsSincePunc := 0
		for _, w := range words {
			b.WriteString(w)
			b.WriteString(" ")
			wordsSincePunc++

			hasPunc := strings.ContainsAny(w, ".,!?;:")
			if hasPunc {
				wordsSincePunc = 0
			} else if wordsSincePunc > 18 {
				b.WriteString(", ") // Inject a comma "breath"
				wordsSincePunc = 0
			}
		}
		text = strings.TrimSpace(b.String())
	}

	return text
}

func squashRepeats(s string, maxRepeats int) string {
	if len(s) == 0 {
		return s
	}
	var b strings.Builder
	runes := []rune(s)
	count := 1
	b.WriteRune(runes[0])
	for i := 1; i < len(runes); i++ {
		if runes[i] == runes[i-1] {
			count++
		} else {
			count = 1
		}
		if count <= maxRepeats {
			b.WriteRune(runes[i])
		}
	}
	return b.String()
}
