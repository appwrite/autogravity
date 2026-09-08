const centerPortrait = '/face-priority/clear-portrait.jpg'
const multiplePortraits = '/face-priority/multiple-faces.jpg'
const blurredPortrait = '/face-priority/blurred-face.jpg'

type Tone = 'before' | 'face' | 'fallback'

type Point = {
  x: number
  y: number
}

type FaceBox = {
  minX: number
  minY: number
  maxX: number
  maxY: number
  primary?: boolean
}

type Frame = {
  phase: 'Before' | 'After'
  strategy: string
  source: 'saliency' | 'face'
  point: Point
  confidence: number
  tone: Tone
  alt: string
  faces?: FaceBox[]
}

type Comparison = {
  id: string
  image: string
  eyebrow: string
  title: string
  description: string
  result: string
  before: Frame
  after: Frame
}

const comparisons: Comparison[] = [
  {
    id: 'clear-portrait',
    image: centerPortrait,
    eyebrow: 'Clear portrait',
    title: 'A face outranks clothing',
    description:
      'Saliency settles on the subject’s torso. Face priority moves the crop anchor 34% of the image height upward.',
    result: '−33.7% y',
    before: {
      phase: 'Before',
      strategy: 'Saliency only',
      source: 'saliency',
      point: { x: 0.4760418385, y: 0.6078201235 },
      confidence: 1,
      tone: 'before',
      alt: 'Centered portrait with the saliency-only focal point on the subject’s torso',
    },
    after: {
      phase: 'After',
      strategy: 'Face priority',
      source: 'face',
      point: { x: 0.4849133417, y: 0.2705245813 },
      confidence: 0.9316979048,
      tone: 'face',
      alt: 'The same centered portrait with the face-priority focal point on the subject’s face',
      faces: [
        {
          minX: 0.4216492542,
          minY: 0.1171214063,
          maxX: 0.5481774292,
          maxY: 0.4239277563,
          primary: true,
        },
      ],
    },
  },
  {
    id: 'multiple-faces',
    image: multiplePortraits,
    eyebrow: 'Multiple faces',
    title: 'The most prominent face wins',
    description:
      'Both faces clear the threshold. Area-weighted priority selects the larger foreground face.',
    result: '2 faces detected',
    before: {
      phase: 'Before',
      strategy: 'Saliency only',
      source: 'saliency',
      point: { x: 0.3159505554, y: 0.6147700424 },
      confidence: 1,
      tone: 'before',
      alt: 'Portrait with two people and the saliency-only focal point on the foreground subject’s torso',
    },
    after: {
      phase: 'After',
      strategy: 'Face priority',
      source: 'face',
      point: { x: 0.318769598, y: 0.3367853218 },
      confidence: 0.9424964754,
      tone: 'face',
      alt: 'The same two-person portrait with both faces detected and the foreground face selected',
      faces: [
        {
          minX: 0.233408108,
          minY: 0.1396942124,
          maxX: 0.404131088,
          maxY: 0.5338764311,
          primary: true,
        },
        {
          minX: 0.7376734428,
          minY: 0.2746350121,
          maxX: 0.8044086017,
          maxY: 0.4290162492,
        },
      ],
    },
  },
  {
    id: 'blurred-face',
    image: blurredPortrait,
    eyebrow: 'No reliable face',
    title: 'Fallback stays intact',
    description:
      'A strongly blurred face does not pass the detector threshold, so U²-Net supplies the same saliency point as before.',
    result: '0 faces detected',
    before: {
      phase: 'Before',
      strategy: 'Saliency only',
      source: 'saliency',
      point: { x: 0.4884092059, y: 0.6575083137 },
      confidence: 1,
      tone: 'before',
      alt: 'Portrait with a strongly blurred face and its saliency focal point',
    },
    after: {
      phase: 'After',
      strategy: 'Saliency fallback',
      source: 'saliency',
      point: { x: 0.4884092059, y: 0.6575083137 },
      confidence: 1,
      tone: 'fallback',
      alt: 'The same blurred portrait with the unchanged saliency fallback point',
    },
  },
]

function percentage(value: number) {
  return `${value * 100}%`
}

function coordinate(value: number) {
  return value.toFixed(4)
}

function FaceBounds({ box }: { box: FaceBox }) {
  return (
    <span
      className={`face-priority-box${box.primary ? ' is-primary' : ''}`}
      style={{
        left: percentage(box.minX),
        top: percentage(box.minY),
        width: percentage(box.maxX - box.minX),
        height: percentage(box.maxY - box.minY),
      }}
      aria-hidden="true"
    >
      <span className="face-priority-box-label">
        {box.primary ? 'primary' : 'detected'}
      </span>
    </span>
  )
}

function FocusMarker({ point, tone }: { point: Point; tone: Tone }) {
  return (
    <svg
      className={`face-priority-marker face-priority-marker--${tone}`}
      style={{ left: percentage(point.x), top: percentage(point.y) }}
      viewBox="0 0 48 48"
      aria-hidden="true"
    >
      <g fill="none" stroke="white" strokeLinecap="round" strokeWidth="5">
        <circle cx="24" cy="24" r="11" />
        <path d="M24 3v10M24 35v10M3 24h10M35 24h10" />
      </g>
      <g fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="2.5">
        <circle cx="24" cy="24" r="11" />
        <path d="M24 3v10M24 35v10M3 24h10M35 24h10" />
      </g>
      <circle cx="24" cy="24" r="2.5" fill="currentColor" stroke="white" strokeWidth="1" />
    </svg>
  )
}

function ComparisonFrame({ image, frame }: { image: string; frame: Frame }) {
  return (
    <figure className={`face-priority-frame face-priority-frame--${frame.tone}`}>
      <div className="face-priority-frame-header">
        <span className="face-priority-phase">
          <span className="face-priority-status-dot" aria-hidden="true" />
          {frame.phase}
        </span>
        <span className="face-priority-strategy">{frame.strategy}</span>
      </div>
      <div className="face-priority-visual">
        <img src={image} alt={frame.alt} loading="lazy" width={800} height={450} />
        {frame.faces?.map((face, index) => (
          <FaceBounds key={`${face.minX}-${index}`} box={face} />
        ))}
        <FocusMarker point={frame.point} tone={frame.tone} />
      </div>
      <figcaption className="face-priority-frame-caption">
        <span className="face-priority-coordinate">
          <span>x</span> {coordinate(frame.point.x)}
        </span>
        <span className="face-priority-coordinate">
          <span>y</span> {coordinate(frame.point.y)}
        </span>
        <span className="face-priority-source">
          source <strong>{frame.source}</strong>
        </span>
        <span className="sr-only">Confidence {coordinate(frame.confidence)}</span>
      </figcaption>
    </figure>
  )
}

export default function FacePriorityDemo() {
  return (
    <div className="face-priority-demo">
      {comparisons.map((comparison, index) => (
        <article
          id={`face-priority-${comparison.id}`}
          className="face-priority-case"
          key={comparison.id}
        >
          <div className="face-priority-case-header">
            <span className="face-priority-case-index" aria-hidden="true">
              {String(index + 1).padStart(2, '0')}
            </span>
            <div className="face-priority-case-copy">
              <span className="face-priority-case-eyebrow">{comparison.eyebrow}</span>
              <h3>{comparison.title}</h3>
              <p>{comparison.description}</p>
            </div>
            <span className="face-priority-result">{comparison.result}</span>
          </div>
          <div className="face-priority-grid">
            <ComparisonFrame image={comparison.image} frame={comparison.before} />
            <ComparisonFrame image={comparison.image} frame={comparison.after} />
          </div>
        </article>
      ))}
    </div>
  )
}
