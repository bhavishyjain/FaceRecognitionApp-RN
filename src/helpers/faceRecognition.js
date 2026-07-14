import { Tensor } from 'onnxruntime-react-native';
import jpeg from 'jpeg-js';
import { toByteArray } from 'base64-js';
import { Platform } from 'react-native';

export const ARCFACE_SIZE = 112;
export const EMBED_DIM = 512;
export const MATCH_THRESHOLD = 0.6;

export function preprocessFaceRGBA(yuvOrRgba, frameW, frameH, faceOrBx, by, bw, bh, targetBuffer) {
  'worklet';
  const OUT = ARCFACE_SIZE;
  
  let cx, cy, cropSize, angle = 0;
  let actualTargetBuffer = targetBuffer;

  if (typeof faceOrBx === 'object' && faceOrBx !== null) {
    const face = faceOrBx;
    actualTargetBuffer = by;
    
    if (face.bounds) {
      cx = face.bounds.x + face.bounds.width / 2;
      cy = face.bounds.y + face.bounds.height / 2;
      cropSize = Math.max(face.bounds.width, face.bounds.height) * 1.4;
      
      if (face.landmarks && face.landmarks.LEFT_EYE && face.landmarks.RIGHT_EYE) {
        const lx = face.landmarks.LEFT_EYE.x;
        const ly = face.landmarks.LEFT_EYE.y;
        const rx = face.landmarks.RIGHT_EYE.x;
        const ry = face.landmarks.RIGHT_EYE.y;
        
        const eyeLeftX = Math.min(lx, rx);
        const eyeLeftY = lx < rx ? ly : ry;
        const eyeRightX = Math.max(lx, rx);
        const eyeRightY = lx < rx ? ry : ly;

        angle = Math.atan2(eyeRightY - eyeLeftY, eyeRightX - eyeLeftX);
      }
    } else {
      cx = frameW / 2;
      cy = frameH / 2;
      cropSize = Math.min(frameW, frameH) * 0.7;
    }
  } else {
    const bx = faceOrBx;
    cx = bx + bw / 2;
    cy = by + bh / 2;
    cropSize = Math.max(bw, bh);
    angle = 0;
  }

  const tensor = actualTargetBuffer || new Float32Array(OUT * OUT * 3);

  const rOff = 0;
  const gOff = OUT * OUT;
  const bOff = OUT * OUT * 2;

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  
  const isGrayscale = yuvOrRgba.length < frameW * frameH * 2;
  const isIOS = Platform.OS === 'ios';

  const stepSize = cropSize / OUT;
  const dRx = stepSize * cosA;
  const dRy = stepSize * sinA;

  let sumY = 0;
  let totalEdgeDiff = 0;
  let edgeDiffCount = 0;

  for (let oy = 0; oy < OUT; oy++) {
    const rowY = (oy / OUT - 0.5) * cropSize;

    const colX0 = -0.5 * cropSize;
    let rx = colX0 * cosA - rowY * sinA;
    let ry = colX0 * sinA + rowY * cosA;

    for (let ox = 0; ox < OUT; ox++) {
      const ix = cx + rx;
      const iy = cy + ry;

      const ix0 = Math.min(Math.max(0, Math.floor(ix)), frameW - 2);
      const iy0 = Math.min(Math.max(0, Math.floor(iy)), frameH - 2);
      const ix1 = ix0 + 1;
      const iy1 = iy0 + 1;
      
      const dx = Math.min(Math.max(0, ix - ix0), 1);
      const dy = Math.min(Math.max(0, iy - iy0), 1);

      let r, g, b;

      if (isGrayscale) {
        const row0 = iy0 * frameW;
        const row1 = iy1 * frameW;
        const y00 = yuvOrRgba[row0 + ix0];
        const y10 = yuvOrRgba[row0 + ix1];
        const y01 = yuvOrRgba[row1 + ix0];
        const y11 = yuvOrRgba[row1 + ix1];

        const Y =
          y00 * (1 - dx) * (1 - dy) +
          y10 * dx * (1 - dy) +
          y01 * (1 - dx) * dy +
          y11 * dx * dy;

        r = Y;
        g = Y;
        b = Y;
      } else {
        const row0 = iy0 * frameW;
        const row1 = iy1 * frameW;
        const i00 = (row0 + ix0) * 4;
        const i10 = (row0 + ix1) * 4;
        const i01 = (row1 + ix0) * 4;
        const i11 = (row1 + ix1) * 4;

        if (isIOS) {
          b =
            yuvOrRgba[i00] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10] * dx * (1 - dy) +
            yuvOrRgba[i01] * (1 - dx) * dy +
            yuvOrRgba[i11] * dx * dy;
          g =
            yuvOrRgba[i00 + 1] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10 + 1] * dx * (1 - dy) +
            yuvOrRgba[i01 + 1] * (1 - dx) * dy +
            yuvOrRgba[i11 + 1] * dx * dy;
          r =
            yuvOrRgba[i00 + 2] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10 + 2] * dx * (1 - dy) +
            yuvOrRgba[i01 + 2] * (1 - dx) * dy +
            yuvOrRgba[i11 + 2] * dx * dy;
        } else {
          r =
            yuvOrRgba[i00] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10] * dx * (1 - dy) +
            yuvOrRgba[i01] * (1 - dx) * dy +
            yuvOrRgba[i11] * dx * dy;
          g =
            yuvOrRgba[i00 + 1] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10 + 1] * dx * (1 - dy) +
            yuvOrRgba[i01 + 1] * (1 - dx) * dy +
            yuvOrRgba[i11 + 1] * dx * dy;
          b =
            yuvOrRgba[i00 + 2] * (1 - dx) * (1 - dy) +
            yuvOrRgba[i10 + 2] * dx * (1 - dy) +
            yuvOrRgba[i01 + 2] * (1 - dx) * dy +
            yuvOrRgba[i11 + 2] * dx * dy;
        }
      }

      const Y_luminance = isGrayscale ? r : (0.299 * r + 0.587 * g + 0.114 * b);
      sumY += Y_luminance;

      const idx = oy * OUT + ox;
      tensor[rOff + idx] = r / 127.5 - 1.0;
      tensor[gOff + idx] = g / 127.5 - 1.0;
      tensor[bOff + idx] = b / 127.5 - 1.0;

      if (ox > 0) {
        const prevIdx = idx - 1;
        const prevY = isGrayscale
          ? (tensor[rOff + prevIdx] + 1.0) * 127.5
          : (0.299 * (tensor[rOff + prevIdx] + 1.0) * 127.5 +
             0.587 * (tensor[gOff + prevIdx] + 1.0) * 127.5 +
             0.114 * (tensor[bOff + prevIdx] + 1.0) * 127.5);
        totalEdgeDiff += Math.abs(Y_luminance - prevY);
        edgeDiffCount++;
      }

      rx += dRx;
      ry += dRy;
    }
  }

  const meanY = sumY / (OUT * OUT);
  const avgEdgeDiff = totalEdgeDiff / (edgeDiffCount || 1);

  tensor.meanLuminance = meanY;
  tensor.isBlurry = avgEdgeDiff < 4.0;

  return tensor;
}

export async function getFaceEmbedding(float32Input, session) {
  const rawEmbed = await runArcFace(session, float32Input);
  return l2normalize(rawEmbed);
}

export function getFaceCropRect(face, frameW, frameH, photoW, photoH) {
  if (!face || !face.bounds) {
    const faceSpan = Math.min(photoW, photoH) * 0.7;
    return {
      bx: photoW / 2 - faceSpan / 2,
      by: photoH / 2 - faceSpan / 2,
      bw: faceSpan,
      bh: faceSpan,
    };
  }

  const { x, y, width, height } = face.bounds;

  const isAligned = (photoW > photoH) === (frameW > frameH);

  let bx, by, bw, bh;

  if (isAligned) {
    const scaleX = photoW / frameW;
    const scaleY = photoH / frameH;
    bx = x * scaleX;
    by = y * scaleY;
    bw = width * scaleX;
    bh = height * scaleY;
  } else {
    const scaleX = photoW / frameH;
    const scaleY = photoH / frameW;
    bx = (frameH - y - height) * scaleX;
    by = x * scaleY;
    bw = height * scaleX;
    bh = width * scaleY;
  }

  const paddingX = bw * 0.2;
  const paddingY = bh * 0.2;

  const paddedBx = bx - paddingX;
  const paddedBy = by - paddingY;
  const paddedBw = bw + 2 * paddingX;
  const paddedBh = bh + 2 * paddingY;

  return { bx: paddedBx, by: paddedBy, bw: paddedBw, bh: paddedBh };
}

export async function runArcFace(session, float32Input) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const inputTensor = new Tensor('float32', float32Input, [
    1,
    3,
    ARCFACE_SIZE,
    ARCFACE_SIZE,
  ]);
  const result = await session.run({ [inputName]: inputTensor });
  return result[outputName].data;
}

export function l2normalize(v) {
  let norm = 0;
  for (let i = 0; i < v.length; i++) {
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm) + 1e-10;
  for (let i = 0; i < v.length; i++) {
    v[i] /= norm;
  }
  return v;
}

export function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

export function decodeJpegBase64(base64) {
  const jpegBytes = toByteArray(base64);
  const { width, height, data } = jpeg.decode(jpegBytes, {
    useTArray: true,
    formatAsRGBA: true,
  });
  return { width, height, data };
}
