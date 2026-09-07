type LogoProps = {
  size?: number
  className?: string
}

export default function Logo({ size = 26, className }: LogoProps) {
  return (
    <svg
      viewBox="0 0 64 64"
      width={size}
      height={size}
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <g fill="currentColor" opacity="0.35">
        <circle cx="12" cy="12" r="5" />
        <circle cx="32" cy="12" r="5" />
        <circle cx="52" cy="12" r="5" />
        <circle cx="12" cy="32" r="5" />
        <circle cx="32" cy="32" r="5" />
        <circle cx="12" cy="52" r="5" />
        <circle cx="32" cy="52" r="5" />
        <circle cx="52" cy="52" r="5" />
      </g>
      <circle cx="52" cy="32" r="10" fill="#fd366e" />
    </svg>
  )
}
