import { describe, expect, it } from "vitest";
import { activeAt, groupWords, validateWord, wrapLines, type Style, type Word } from "./captions";

const base = { mode: "phrase", max_words: 4, max_chars: 28, max_lines: 2, words_visible: 1 } as Style;
const w = (text: string, start: number, end: number): Word => ({ text, start, end, kind: "word" });
const words = [w("Hello", 0.2, 0.6), w("world.", 0.7, 1.2), w("This", 1.4, 1.7), w("is", 1.75, 1.9)];

describe("groupWords", () => {
  it("shows one word at a time in word mode", () => {
    expect(groupWords(words, { ...base, mode: "word" })).toHaveLength(4);
  });
  it("breaks phrases at sentence ends", () => {
    const g = groupWords(words, { ...base, max_words: 10 });
    expect(g[0].words.map((x) => x.text)).toEqual(["Hello", "world."]);
  });
  it("breaks on long pauses", () => {
    expect(groupWords([w("a", 0, 0.5), w("b", 3, 3.5)], { ...base, max_words: 10 })).toHaveLength(2);
  });
  it("keeps Kannada text intact", () => {
    const g = groupWords([w("ನಮಸ್ಕಾರ", 0, 1), w("ಕನ್ನಡ", 1, 2)], { ...base, max_words: 5 });
    expect(g[0].words.map((x) => x.text).join(" ")).toBe("ನಮಸ್ಕಾರ ಕನ್ನಡ");
  });
});

describe("activeAt", () => {
  it("finds the caption and the spoken word", () => {
    const g = groupWords(words, { ...base, max_words: 10 });
    expect(activeAt(g, 0.8, 0)?.index).toBe(1);
    expect(activeAt(g, 0.0, 0)).toBeNull();
  });
  it("applies the offset", () => {
    const g = groupWords(words, { ...base, max_words: 10 });
    expect(activeAt(g, 0.8, 0.5)?.index).toBe(0);
  });
});

describe("validateWord", () => {
  it("rejects bad timing", () => {
    expect(validateWord(w("a", 2, 1), 10)).toMatch(/after start/);
    expect(validateWord(w("a", -1, 1), 10)).toMatch(/negative/);
    expect(validateWord(w("", 0, 1), 10)).toMatch(/empty/);
    expect(validateWord(w("a", 0, 1), 10)).toBeNull();
  });
});

describe("wrapLines", () => {
  it("never loses words", () => {
    expect(wrapLines("one two three four five six", 8, 2).join(" ")).toBe("one two three four five six");
  });
});
