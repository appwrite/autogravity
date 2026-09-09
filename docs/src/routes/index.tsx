import { createFileRoute } from '@tanstack/react-router'
import CodePanel from '../components/CodePanel'
import DocsSidebar from '../components/DocsSidebar'
import FacePriorityDemo from '../components/FacePriorityDemo'
import GravityDemo from '../components/GravityDemo'
import HttpEndpoint from '../components/HttpEndpoint'
import InlineCode from '../components/InlineCode'
import SectionTitle from '../components/SectionTitle'

export const Route = createFileRoute('/')({ component: DocsPage })

function DocsPage() {
  return (
    <div className="page-grid">
      <DocsSidebar />

      <main className="doc-main">
        <section id="overview" className="doc-section doc-section--hero">
          <span className="section-kicker">Overview</span>
          <h1 className="doc-h1">Focal points, as a service.</h1>
          <p className="doc-lead">
            A small Go HTTP service that finds the best crop focus in an image.
            It prioritizes confidently detected faces with YuNet, then falls
            back to U²-Net saliency. It never crops, stores, identifies, or
            modifies the submitted image.
          </p>
          <div className="pill-row">
            <span className="pill">Face-first · saliency fallback</span>
            <span className="pill">JPEG · PNG · WebP</span>
            <span className="pill">CPU-only</span>
          </div>
        </section>

        <section id="face-priority" className="doc-section">
          <SectionTitle
            kicker="How it works"
            title="Face-first, with a safe fallback"
            lead="The current pipeline runs YuNet first. A reliable face becomes the focal point immediately; without one, the same U²-Net saliency path continues unchanged. These comparisons use real model output from the integration fixtures."
          />
          <FacePriorityDemo />
          <p className="doc-footnote">
            Green boxes are detected faces, the thicker box is the selected
            primary face, and each crosshair is the returned normalized gravity
            coordinate. No identity recognition is performed.
          </p>
        </section>

        <section id="preview" className="doc-section">
          <SectionTitle
            kicker="Integration"
            title="Storage preview"
            lead="Appwrite Storage accepts gravity=auto on the preview endpoint. When both width and height are set, autogravity finds the main subject and crops around its focal point instead of the geometric centre."
          />
          <GravityDemo />
          <HttpEndpoint
            method="GET"
            path="/v1/storage/buckets/{bucketId}/files/{fileId}/preview"
            description="Pass width, height, and gravity=auto to crop around the detected focal point."
          />
          <CodePanel label="Example">
            <code>
              ?width=400&amp;height=400&amp;gravity=auto
            </code>
          </CodePanel>
        </section>

        <section id="install" className="doc-section">
          <SectionTitle
            kicker="Start"
            title="Install"
            lead="Fastest path is Docker. The image includes verified YuNet and U²-Net models plus the CPU-only ONNX Runtime library."
          />
          <CodePanel label="Docker">
            <code>
              <span className="prompt">$ </span>docker run --rm -p 8080:8080{' '}
              ghcr.io/appwrite/autogravity
              {'\n'}
              <span className="prompt">$ </span>docker build -t autogravity .
              {'\n'}
              <span className="prompt">$ </span>docker run --rm -p 8080:8080
              autogravity
            </code>
          </CodePanel>
          <p className="doc-copy">
            Or build from source with Go 1.25 or newer.{' '}
            <InlineCode>make model</InlineCode> downloads and verifies the ONNX
            models.
          </p>
          <CodePanel label="Source">
            <code>
              <span className="prompt">$ </span>make model
              {'\n'}
              <span className="prompt">$ </span>make build
              {'\n'}
              <span className="prompt">$ </span>./autogravity
            </code>
          </CodePanel>
        </section>

        <section id="config" className="doc-section">
          <SectionTitle kicker="Start" title="Configuration" />
          <div className="data-table">
            <div className="data-table-head">
              <span>Variable</span>
              <span>Default</span>
              <span>Purpose</span>
            </div>
            <div className="data-table-row">
              <span className="accent">ADDR</span>
              <span className="accent">:8080</span>
              <span className="data-table-desc">HTTP listen address</span>
            </div>
            <div className="data-table-row">
              <span className="accent">MODEL_PATH</span>
              <span className="accent">unset</span>
              <span className="data-table-desc">U²-Net model path</span>
            </div>
            <div className="data-table-row">
              <span className="accent">FACE_MODEL_PATH</span>
              <span className="accent">models/face_detection_yunet_2023mar.onnx</span>
              <span className="data-table-desc">YuNet model path</span>
            </div>
            <div className="data-table-row">
              <span className="accent">FACE_SCORE_THRESHOLD</span>
              <span className="accent">0.85</span>
              <span className="data-table-desc">Minimum reliable face score</span>
            </div>
            <div className="data-table-row">
              <span className="accent">ONNXRUNTIME_LIB</span>
              <span className="required">required</span>
              <span className="data-table-desc">
                Full ONNX Runtime shared-library path
              </span>
            </div>
          </div>
        </section>

        <section id="api" className="doc-section">
          <SectionTitle
            kicker="Reference"
            title="API"
            lead="Send a JPEG, PNG, or WebP image as a multipart image field, or as the raw request body."
          />

          <HttpEndpoint
            method="POST"
            path="/analyze"
            description="Returns a prioritized face center or the strongest salient region's weighted focal point."
          />

          <div className="code-grid">
            <CodePanel label="Request">
              <code>
                curl -sS -X POST \{'\n'}
                {'  '}http://localhost:8080/analyze \{'\n'}
                {'  '}-F <span className="str">'image=@photo.jpg'</span>
              </code>
            </CodePanel>
            <CodePanel label="Response">
              <code>
                {'{'}{'\n'}
                {'  '}<span className="key">"gravity"</span>: {'{'}{'\n'}
                {'    '}<span className="key">"x"</span>:{' '}
                <span className="num">0.68</span>,{'\n'}
                {'    '}<span className="key">"y"</span>:{' '}
                <span className="num">0.37</span>{'\n'}
                {'  '}{'}'},{'\n'}
                {'  '}<span className="key">"confidence"</span>:{' '}
                <span className="num">0.91</span>,{'\n'}
                {'  '}<span className="key">"source"</span>:{' '}
                <span className="str">"face"</span>{'\n'}
                {'}'}
              </code>
            </CodePanel>
          </div>

          <div className="diagram-card">
            <svg viewBox="0 0 200 120" width="200" height="120" aria-hidden="true">
              <rect width="200" height="120" fill="var(--raised)" />
              <line
                x1="136"
                y1="0"
                x2="136"
                y2="120"
                stroke="#fd366e"
                strokeWidth="1"
                opacity="0.5"
              />
              <line
                x1="0"
                y1="44.4"
                x2="200"
                y2="44.4"
                stroke="#fd366e"
                strokeWidth="1"
                opacity="0.5"
              />
              <circle cx="136" cy="44.4" r="6" fill="#fd366e" />
              <text
                x="10"
                y="112"
                fill="var(--muted)"
                fontFamily="JetBrains Mono, monospace"
                fontSize="10"
              >
                1280 × 720
              </text>
            </svg>
            <p className="doc-copy doc-copy--flush">
              Coordinates are in <InlineCode>[0.0, 1.0]</InlineCode>, measured
              from the oriented image&apos;s top-left corner. EXIF orientation is
              applied before analysis. A reliable face supplies its bounding-box
              center; otherwise U²-Net supplies the saliency centroid. The{' '}
              <InlineCode>source</InlineCode> field identifies which strategy was
              selected. Confidence is that strategy&apos;s model score, not an
              identity match or a calibrated probability.
            </p>
          </div>

          <HttpEndpoint
            method="GET"
            path="/healthz"
            description='Returns 503 until the models are loaded, then 200 with {"status":"ok"}.'
          />
        </section>

        <section id="limits" className="doc-section">
          <SectionTitle kicker="Reference" title="Limits" />
          <p className="doc-copy">
            Requests are limited to 10 MiB and decoded images to 20 megapixels.
            Separate upload and analysis admission limits bound buffered-body and
            decoded-image memory without allowing slow uploads to reserve
            inference capacity. Both models are loaded once at startup and their
            inference sessions are reused across requests.
          </p>
        </section>

        <section id="performance" className="doc-section">
          <SectionTitle kicker="Reference" title="Performance" />
          <div className="data-table performance">
            <div className="data-table-head">
              <span>Input</span>
              <span>Dimensions</span>
              <span>Time per image</span>
            </div>
            <div className="data-table-row">
              <span className="data-table-label">Landscape JPEG</span>
              <span className="accent">1280 × 720</span>
              <span className="accent">391.4 ms</span>
            </div>
            <div className="data-table-row">
              <span className="data-table-label">Portrait PNG</span>
              <span className="accent">720 × 1080</span>
              <span className="accent">291.3 ms</span>
            </div>
          </div>
          <p className="doc-footnote">
            Historical U²-Net fallback timings on Apple M3 Pro, CPU-only ONNX
            Runtime 1.23.2, Go 1.25.14. Median of five sequential benchmark
            samples. Face-selected requests skip U²-Net. Performance varies with
            hardware and input images.
          </p>
        </section>
      </main>
    </div>
  )
}
