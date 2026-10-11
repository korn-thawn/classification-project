// Every image (inspection and training sample) goes through normalizeImage so
// the Python side always sees the same resolution and orientation; the ROI in
// config.yaml is in pixels of this normalized frame.
export const MAX_SIDE = 1280; // keep equal to camera.width in config.yaml
const JPEG_QUALITY = 0.92;

/** Apply EXIF rotation, scale the long side to MAX_SIDE and re-encode as JPEG. */
export async function normalizeImage(blob, maxSide = MAX_SIDE) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const scale = maxSide / Math.max(bmp.width, bmp.height);
  const width = Math.round(bmp.width * scale);
  const height = Math.round(bmp.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, width, height);
  bmp.close();
  const out = await new Promise((ok) => canvas.toBlob(ok, "image/jpeg", JPEG_QUALITY));
  return { blob: out, width, height };
}

/** Grab the current frame of a <video> element as a PNG blob. */
export function grabFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((ok) => canvas.toBlob(ok, "image/png"));
}
