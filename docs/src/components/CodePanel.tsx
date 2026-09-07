import type { ReactNode } from 'react'

type CodePanelProps = {
  label: string
  children: ReactNode
}

export default function CodePanel({ label, children }: CodePanelProps) {
  return (
    <div className="code-panel">
      <div className="code-panel-header">
        <span className="code-panel-label">{label}</span>
      </div>
      <pre className="code-block code-block--panel">{children}</pre>
    </div>
  )
}
