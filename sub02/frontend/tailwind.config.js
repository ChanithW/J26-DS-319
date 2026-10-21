/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        accent: { DEFAULT: "#4f46e5", dark: "#818cf8" },
      },
    },
  },
  plugins: [],
};
