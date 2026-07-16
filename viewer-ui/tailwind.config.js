/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#18212f",
        muted: "#667385",
        line: "#dde3ea",
        soft: "#f6f7f9",
      },
      boxShadow: {
        panel: "0 18px 40px rgba(24, 33, 47, 0.07)",
      },
    },
  },
  plugins: [],
};
