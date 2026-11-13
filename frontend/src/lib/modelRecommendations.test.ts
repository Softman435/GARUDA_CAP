import { describe, expect, it } from "vitest";
import { recommendedModel } from "./modelRecommendations";

const models = { "kn-medium": "Kannada fine-tune", "large-v3": "Multilingual", base: "Fast" };

describe("recommendedModel", () => {
  it.each(["kn", "kanglish"])("uses the Kannada-tuned model for %s", (language) => {
    expect(recommendedModel(language, models, "base")).toBe("kn-medium");
  });

  it.each(["en", "te", "kn-en", "auto"])("uses the accurate multilingual model for %s", (language) => {
    expect(recommendedModel(language, models, "base")).toBe("large-v3");
  });

  it("falls back to the configured default when the recommendation is unavailable", () => {
    expect(recommendedModel("kn", { base: "Fast" }, "base")).toBe("base");
  });
});
