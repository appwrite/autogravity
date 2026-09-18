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
            The default backend prioritizes confidently detected faces with
            YuNet, then falls back to U²-Net saliency.{' '}
            <InlineCode>MODEL_BACKEND=focalnet</InlineCode> switches to
            Appwrite&apos;s own compact FocalNet model. It never crops, stores,
            identifies, or modifies the submitted image.
          </p>
          <div className="pill-row">
            <span className="pill">Face-first · saliency fallback</span>
            <span className="pill">FocalNet backend</span>
            <span className="pill">JPEG · PNG · WebP · GIF</span>
            <span className="pill">CPU-only</span>
          </div>
        </section>

        <section id="face-priority" className="doc-section">
          <SectionTitle
            kicker="How it works"
            title="Face-first, with a safe fallback"
            lead="The default backend runs YuNet first. A reliable face becomes the focal point immediately; without one, the same U²-Net saliency path continues unchanged. These comparisons use real model output from the integration fixtures."
          />
          <FacePriorityDemo />
          <p className="doc-footnote">
            Green boxes are detected faces, the thicker box is the selected
            primary face, and each crosshair is the returned normalized gravity
            coordinate. No identity recognition is performed.
          </p>
        </section>

        <section id="focalnet" className="doc-section">
          <SectionTitle
            kicker="How it works"
            title="Appwrite's FocalNet model"
            lead="FocalNet distills Autogravity's YuNet + U²-Net teacher into one 19 MiB FP32 ONNX. It predicts a 64×64 importance map, ranks candidate crops with a human-preference head, and returns the selected crop center. Faces are fused into the map; YuNet is not run on this path."
          />
          <p className="doc-copy">
            The default production backend stays YuNet + U²-Net. Set{' '}
            <InlineCode>MODEL_BACKEND=focalnet</InlineCode> after{' '}
            <InlineCode>make model-focalnet</InlineCode> to load the public{' '}
            <InlineCode>2026-09-14-rc1</InlineCode> weights from{' '}
            <a
              href="https://github.com/appwrite/focalnet"
              target="_blank"
              rel="noreferrer"
            >
              appwrite/focalnet
            </a>
            . Pass <InlineCode>?aspect_ratio=16:9</InlineCode> to rank a
            non-square crop; the default is <InlineCode>1:1</InlineCode>.
          </p>
          <CodePanel label="Run FocalNet">
            <code>
              <span className="prompt">$ </span>make model-focalnet
              {'\n'}
              <span className="prompt">$ </span>MODEL_BACKEND=focalnet
              ./autogravity
              {'\n'}
              <span className="prompt">$ </span>docker run --rm -p 8080:8080 \
              {'\n'}
              {'    '}-e MODEL_BACKEND=focalnet \
              {'\n'}
              {'    '}ghcr.io/appwrite/autogravity
            </code>
          </CodePanel>
          <div className="data-table">
            <div className="data-table-head">
              <span>Backend</span>
              <span>Models</span>
              <span>Result</span>
            </div>
            <div className="data-table-row">
              <span className="accent">u2net</span>
              <span className="accent">YuNet + U²-Net</span>
              <span className="data-table-desc">
                Face center, or saliency centroid. Default.
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">focalnet</span>
              <span className="accent">focalnet-human.onnx</span>
              <span className="data-table-desc">
                Ranked crop center plus crop rectangle. Optional.
              </span>
            </div>
          </div>
          <p className="doc-footnote">
            FocalNet reports map MAE 0.09975 and 91.09% 1:1 importance retained
            against the teacher on a 10k validation split. Those figures are
            teacher-agreement, not a guarantee that Autogravity traffic will
            match the face-priority backend.
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
            lead="Fastest path is Docker. The image includes verified YuNet and U²-Net models plus the CPU-only ONNX Runtime library. Release images also bundle FocalNet; PR images may omit it."
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
            <InlineCode>make model</InlineCode> downloads and verifies the
            default ONNX models.{' '}
            <InlineCode>make model-focalnet</InlineCode> downloads Appwrite&apos;s
            FocalNet weights.
          </p>
          <CodePanel label="Source">
            <code>
              <span className="prompt">$ </span>make model
              {'\n'}
              <span className="prompt">$ </span>make build
              {'\n'}
              <span className="prompt">$ </span>./autogravity
              {'\n'}
              <span className="prompt">$ </span>make model-focalnet
              {'\n'}
              <span className="prompt">$ </span>MODEL_BACKEND=focalnet
              ./autogravity
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
              <span className="accent">MODEL_BACKEND</span>
              <span className="accent">u2net</span>
              <span className="data-table-desc">
                u2net (YuNet + U²-Net) or focalnet (Appwrite&apos;s model)
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">MODEL_PRECISION</span>
              <span className="accent">int8</span>
              <span className="data-table-desc">
                int8 or fp32 for the U²-Net backend
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">MODEL_PATH</span>
              <span className="accent">unset</span>
              <span className="data-table-desc">
                Explicit ONNX path; overrides precision / FocalNet default
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">FACE_MODEL_PATH</span>
              <span className="accent">models/face_detection_yunet_2023mar.onnx</span>
              <span className="data-table-desc">
                YuNet model path (U²-Net backend only)
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">FACE_SCORE_THRESHOLD</span>
              <span className="accent">0.85</span>
              <span className="data-table-desc">
                Minimum reliable face score (U²-Net backend only)
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">ONNXRUNTIME_LIB</span>
              <span className="required">required</span>
              <span className="data-table-desc">
                Full ONNX Runtime shared-library path
              </span>
            </div>
            <div className="data-table-row">
              <span className="accent">MAX_REQUEST_SIZE</span>
              <span className="accent">10MiB</span>
              <span className="data-table-desc">
                Maximum /analyze request body, including multipart overhead
              </span>
            </div>
          </div>
        </section>

        <section id="api" className="doc-section">
          <SectionTitle
            kicker="Reference"
            title="API"
            lead="Send a JPEG, PNG, WebP, or GIF image as a multipart image field, or as the raw request body."
          />

          <HttpEndpoint
            method="POST"
            path="/analyze"
            description="Returns a prioritized face center, the strongest salient region's weighted focal point, or a FocalNet-ranked crop center."
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
              applied before analysis. On the default backend a reliable face
              supplies its bounding-box center; otherwise U²-Net supplies the
              saliency centroid. Set <InlineCode>MODEL_BACKEND=focalnet</InlineCode>{' '}
              to use Appwrite&apos;s human-ranking model instead.{' '}
              <InlineCode>source</InlineCode> is then{' '}
              <InlineCode>focalnet</InlineCode>, YuNet is not consulted, and the
              response includes a <InlineCode>crop</InlineCode> rectangle.
              Confidence is that strategy&apos;s model score, not an identity
              match or a calibrated probability.
            </p>
          </div>

          <CodePanel label="FocalNet request">
            <code>
              curl -sS -X POST \{'\n'}
              {'  '}http://localhost:8080/analyze?aspect_ratio=16:9 \{'\n'}
              {'  '}-F <span className="str">'image=@photo.jpg'</span>
            </code>
          </CodePanel>
          <CodePanel label="FocalNet response">
            <code>
              {'{'}{'\n'}
              {'  '}<span className="key">"gravity"</span>: {'{'}{'\n'}
              {'    '}<span className="key">"x"</span>:{' '}
              <span className="num">0.52</span>,{'\n'}
              {'    '}<span className="key">"y"</span>:{' '}
              <span className="num">0.41</span>{'\n'}
              {'  '}{'}'},{'\n'}
              {'  '}<span className="key">"confidence"</span>:{' '}
              <span className="num">0.91</span>,{'\n'}
              {'  '}<span className="key">"source"</span>:{' '}
              <span className="str">"focalnet"</span>,{'\n'}
              {'  '}<span className="key">"crop"</span>: {'{'}{'\n'}
              {'    '}<span className="key">"left"</span>:{' '}
              <span className="num">120</span>,{'\n'}
              {'    '}<span className="key">"top"</span>:{' '}
              <span className="num">40</span>,{'\n'}
              {'    '}<span className="key">"width"</span>:{' '}
              <span className="num">480</span>,{'\n'}
              {'    '}<span className="key">"height"</span>:{' '}
              <span className="num">480</span>,{'\n'}
              {'    '}<span className="key">"retained_importance"</span>:{' '}
              <span className="num">0.88</span>{'\n'}
              {'  '}{'}'}{'\n'}
              {'}'}
            </code>
          </CodePanel>

          <HttpEndpoint
            method="GET"
            path="/healthz"
            description='Returns 503 until the models are loaded, then 200 with {"status":"ok"}.'
          />
        </section>

        <section id="limits" className="doc-section">
          <SectionTitle kicker="Reference" title="Limits" />
          <p className="doc-copy">
            Requests default to a 10 MiB body limit (
            <InlineCode>MAX_REQUEST_SIZE</InlineCode>) and decoded images to 48
            megapixels. Set <InlineCode>MAX_REQUEST_SIZE</InlineCode> to a
            positive byte count such as <InlineCode>10MiB</InlineCode> or{' '}
            <InlineCode>10485760</InlineCode>.
            Separate upload and analysis admission limits bound buffered-body and
            decoded-image memory without allowing slow uploads to reserve
            inference capacity. Both models are loaded once at startup and their
            inference sessions are reused across requests. FocalNet loads a
            single session instead of YuNet + U²-Net.
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
            samples. Face-selected requests skip U²-Net. FocalNet is a 19 MiB
            FP32 graph versus ~42 MiB INT8 U²-Net plus YuNet; measure it on
            your deploy target. Performance varies with hardware and input
            images.
          </p>
        </section>
      </main>
    </div>
  )
}
