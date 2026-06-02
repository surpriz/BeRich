import type { Metadata } from "next";
import { Bricolage_Grotesque, Hanken_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { I18nProvider } from "./lib/i18n";
import { LevelProvider } from "./lib/level";
import { StrategyProvider } from "./lib/strategy";
import { HealthFooter } from "./components/HealthFooter";
import { TopNav } from "./components/TopNav";

const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-bricolage",
  weight: ["500", "700", "800"],
});
const hanken = Hanken_Grotesk({
  subsets: ["latin"],
  variable: "--font-hanken",
  weight: ["400", "500", "600"],
});
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "BeRich — swing signals",
  description: "ML-driven swing-trading advice with rigorous backtest validation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${bricolage.variable} ${hanken.variable} ${jetbrains.variable}`}>
      <body>
        <I18nProvider>
          <LevelProvider>
            <StrategyProvider>
              <TopNav />
              {children}
              <HealthFooter />
            </StrategyProvider>
          </LevelProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
