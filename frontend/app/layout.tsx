import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Social Media Risk Classifier',
  description: 'Tweet-style demo powered by DistilBERT',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, background: '#0b0f14', color: '#e7e9ea', minHeight: '100vh' }}>
        {children}
      </body>
    </html>
  )
}
