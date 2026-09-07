type SectionTitleProps = {
  kicker?: string
  title: string
  lead?: string
}

export default function SectionTitle({ kicker, title, lead }: SectionTitleProps) {
  return (
    <div className="section-title">
      {kicker ? <span className="section-kicker">{kicker}</span> : null}
      <h2 className="doc-h2">{title}</h2>
      {lead ? <p className="doc-lead">{lead}</p> : null}
    </div>
  )
}
