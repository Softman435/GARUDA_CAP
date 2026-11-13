/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { ink: "#09090b", panel: "#111116", line: "#332d2c", gold: "#ffd11a", saffron: "#f25031" },
    },
  },
  plugins: [],
};
