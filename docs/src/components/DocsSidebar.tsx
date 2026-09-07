import { useEffect, useState } from 'react'
import { NAV } from '../lib/site'

export default function DocsSidebar() {
  const [active, setActive] = useState('#overview')

  useEffect(() => {
    const ids = [...NAV.start, ...NAV.reference].map((item) => item.href)
    const sections = ids
      .map((id) => document.querySelector(id))
      .filter((el): el is HTMLElement => el instanceof HTMLElement)

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]

        if (visible?.target.id) {
          setActive(`#${visible.target.id}`)
        }
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: [0, 0.25, 0.5, 1] },
    )

    for (const section of sections) {
      observer.observe(section)
    }

    return () => observer.disconnect()
  }, [])

  return (
    <aside className="sidebar" aria-label="Documentation">
      <div className="nav-group">
        <span className="nav-kicker">Start</span>
        {NAV.start.map((item) => (
          <a
            key={item.href}
            href={item.href}
            className={`nav-link${active === item.href ? ' is-active' : ''}`}
          >
            {item.label}
          </a>
        ))}
      </div>
      <div className="nav-group">
        <span className="nav-kicker">Reference</span>
        {NAV.reference.map((item) => (
          <a
            key={item.href}
            href={item.href}
            className={`nav-link${active === item.href ? ' is-active' : ''}`}
          >
            {item.label}
          </a>
        ))}
      </div>
    </aside>
  )
}
