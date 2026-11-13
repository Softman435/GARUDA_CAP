export function recommendedModel(language: string, models: Record<string, string>, fallback: string): string {
  const preferred = language === "kn" || language === "kanglish" ? "kn-medium" : "large-v3";
  if (models[preferred] !== undefined) return preferred;
  if (models[fallback] !== undefined) return fallback;
  return Object.keys(models)[0] ?? fallback;
}
