/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: { sans: ["\"Source Sans 3\"", "\"Source Sans Pro\"", "ui-sans-serif", "system-ui", "sans-serif"] },
      colors: {
        accent: { DEFAULT: "#4f46e5", dark: "#818cf8" },
      },
    },
  },
  plugins: [],
};
