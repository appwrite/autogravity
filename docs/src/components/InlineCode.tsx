type InlineCodeProps = {
  children: React.ReactNode
}

export default function InlineCode({ children }: InlineCodeProps) {
  return <code className="inline-code">{children}</code>
}
