import { storagePreviewUrl, storageViewUrl } from '../lib/storage-demo'
import InlineCode from './InlineCode'

export default function GravityDemo() {
  const originalUrl = storageViewUrl()
  const centerUrl = storagePreviewUrl('center')
  const autoUrl = storagePreviewUrl('auto')

  return (
    <div className="gravity-demo">
      <figure className="gravity-demo-original">
        <img src={originalUrl} alt="Golden retriever sitting on grass in a park" loading="lazy" />
        <figcaption className="gravity-demo-caption">Original</figcaption>
      </figure>

      <div className="gravity-demo-crops">
        <figure className="gravity-demo-crop">
          <img
            src={centerUrl}
            alt="400 by 400 preview cropped with gravity center"
            loading="lazy"
            width={400}
            height={400}
          />
          <figcaption className="gravity-demo-caption">
            <InlineCode>gravity=center</InlineCode>
            <span className="gravity-demo-note">Geometric centre — subject cut off</span>
          </figcaption>
        </figure>

        <figure className="gravity-demo-crop">
          <img
            src={autoUrl}
            alt="400 by 400 preview cropped with gravity auto"
            loading="lazy"
            width={400}
            height={400}
          />
          <figcaption className="gravity-demo-caption">
            <InlineCode>gravity=auto</InlineCode>
            <span className="gravity-demo-note">Focal point — subject kept in frame</span>
          </figcaption>
        </figure>
      </div>
    </div>
  )
}
