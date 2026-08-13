import { Tensor } from 'onnxruntime-react-native';
import { Platform } from 'react-native';

export const ARCFACE_SIZE = 112;
export const EMBED_DIM = 512;
export const MATCH_THRESHOLD = 0.6;
export const LIVENESS_THRESHOLD = 0.8;

export function preprocessFaceRGBA(
  yuvOrRgba,
  frameW,
  frameH,
  faceOrBx,
  by,
  bw,
  bh,
  targetBuffer,
) {
  'worklet';
  const OUT = ARCFACE_SIZE;

  let cx,
    cy,
    cropSize,
    angle = 0;
  let actualTargetBuffer = targetBuffer;

  if (typeof faceOrBx === 'object' && faceOrBx !== null) {
    const face = faceOrBx;
    actualTargetBuffer = by;

    if (face.bounds) {
      cx = face.bounds.x + face.bounds.width / 2;
      cy = face.bounds.y + face.bounds.height / 2;
      cropSize = Math.max(face.bounds.width, face.bounds.height) * 1.4;

      if (
        face.landmarks &&
        face.landmarks.LEFT_EYE &&
        face.landmarks.RIGHT_EYE
      ) {
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

  const isIOS = Platform.OS === 'ios';
  const isYUV =
    !isIOS &&
    yuvOrRgba.length >= frameW * frameH * 1.4 &&
    yuvOrRgba.length < frameW * frameH * 2;
  const isGrayscale = !isIOS && yuvOrRgba.length < frameW * frameH * 1.4;

  const stepSize = cropSize / OUT;
  const dRx = stepSize * cosA;
  const dRy = stepSize * sinA;

  let sumY = 0;
  let totalEdgeDiff = 0;
  let edgeDiffCount = 0;

  let sumR = 0;
  let sumB = 0;

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
      } else if (isYUV) {
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

        const uvRow = Math.floor(iy0 / 2);
        const uvCol = Math.floor(ix0 / 2);
        const uvIndex = frameW * frameH + uvRow * frameW + uvCol * 2;

        let uVal = 128;
        let vVal = 128;
        if (uvIndex + 1 < yuvOrRgba.length) {
          uVal = yuvOrRgba[uvIndex];
          vVal = yuvOrRgba[uvIndex + 1];
        }

        const u = uVal - 128;
        const v = vVal - 128;

        r = Y + 1.402 * v;
        g = Y - 0.344136 * u - 0.714136 * v;
        b = Y + 1.772 * u;

        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));

        sumR += r;
        sumB += b;
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

      const Y_luminance = isGrayscale ? r : 0.299 * r + 0.587 * g + 0.114 * b;
      sumY += Y_luminance;

      const idx = oy * OUT + ox;
      tensor[rOff + idx] = (r - 127.5) / 128.0;
      tensor[gOff + idx] = (g - 127.5) / 128.0;
      tensor[bOff + idx] = (b - 127.5) / 128.0;

      if (ox > 0) {
        const prevIdx = idx - 1;
        const prevY = isGrayscale
          ? (tensor[rOff + prevIdx] + 1.0) * 127.5
          : 0.299 * (tensor[rOff + prevIdx] + 1.0) * 127.5 +
            0.587 * (tensor[gOff + prevIdx] + 1.0) * 127.5 +
            0.114 * (tensor[bOff + prevIdx] + 1.0) * 127.5;
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

  if (isYUV && sumB > sumR) {
    for (let idx = 0; idx < OUT * OUT; idx++) {
      const temp = tensor[rOff + idx];
      tensor[rOff + idx] = tensor[bOff + idx];
      tensor[bOff + idx] = temp;
    }
  }

  return tensor;
}

export async function getFaceEmbedding(float32Input, session) {
  const rawEmbed = await runArcFace(session, float32Input);
  return l2normalize(rawEmbed);
}

export async function runArcFace(session, float32Input) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const hwcInput = new Float32Array(ARCFACE_SIZE * ARCFACE_SIZE * 3);
  const planeSize = ARCFACE_SIZE * ARCFACE_SIZE;

  for (let i = 0; i < planeSize; i++) {
    const out = i * 3;
    hwcInput[out] = float32Input[i];
    hwcInput[out + 1] = float32Input[planeSize + i];
    hwcInput[out + 2] = float32Input[planeSize * 2 + i];
  }

  const inputTensor = new Tensor('float32', hwcInput, [
    1,
    ARCFACE_SIZE,
    ARCFACE_SIZE,
    3,
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

export function preprocessFaceFAS(
  yuvOrRgba,
  frameW,
  frameH,
  face,
  scale,
  targetBuffer,
) {
  'worklet';
  const OUT = 80;

  let centerX, centerY;
  let newWidth, newHeight;
  let actualTargetBuffer = targetBuffer;

  if (face && face.bounds) {
    const x = face.bounds.x;
    const y = face.bounds.y;
    const boxW = face.bounds.width;
    const boxH = face.bounds.height;

    const limitScale = Math.min(
      (frameH - 1) / boxH,
      Math.min((frameW - 1) / boxW, scale),
    );

    newWidth = boxW * limitScale;
    newHeight = boxH * limitScale;
    centerX = boxW / 2 + x;
    centerY = boxH / 2 + y;

    let ltx = centerX - newWidth / 2;
    let lty = centerY - newHeight / 2;
    let rbx = centerX + newWidth / 2;
    let rby = centerY + newHeight / 2;

    if (ltx < 0) {
      rbx -= ltx;
      ltx = 0;
    }
    if (lty < 0) {
      rby -= lty;
      lty = 0;
    }
    if (rbx > frameW - 1) {
      ltx -= rbx - frameW + 1;
      rbx = frameW - 1;
    }
    if (rby > frameH - 1) {
      lty -= rby - frameH + 1;
      rby = frameH - 1;
    }

    centerX = (ltx + rbx) / 2;
    centerY = (lty + rby) / 2;
  } else {
    centerX = frameW / 2;
    centerY = frameH / 2;
    newWidth = frameW * 0.7;
    newHeight = frameH * 0.7;
  }

  const tensor = actualTargetBuffer || new Float32Array(OUT * OUT * 3);

  const bOff = 0;
  const gOff = OUT * OUT;
  const rOff = OUT * OUT * 2;

  const isIOS = Platform.OS === 'ios';
  const isYUV =
    !isIOS &&
    yuvOrRgba.length >= frameW * frameH * 1.4 &&
    yuvOrRgba.length < frameW * frameH * 2;
  const isGrayscale = !isIOS && yuvOrRgba.length < frameW * frameH * 1.4;

  let sumR = 0;
  let sumB = 0;

  for (let oy = 0; oy < OUT; oy++) {
    const localY = (oy / OUT - 0.5) * newHeight;

    for (let ox = 0; ox < OUT; ox++) {
      const localX = (ox / OUT - 0.5) * newWidth;

      const ix = centerX + localX;
      const iy = centerY + localY;

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
        const Y =
          yuvOrRgba[row0 + ix0] * (1 - dx) * (1 - dy) +
          yuvOrRgba[row0 + ix1] * dx * (1 - dy) +
          yuvOrRgba[row1 + ix0] * (1 - dx) * dy +
          yuvOrRgba[row1 + ix1] * dx * dy;
        r = Y;
        g = Y;
        b = Y;
      } else if (isYUV) {
        const row0 = iy0 * frameW;
        const row1 = iy1 * frameW;
        const Y =
          yuvOrRgba[row0 + ix0] * (1 - dx) * (1 - dy) +
          yuvOrRgba[row0 + ix1] * dx * (1 - dy) +
          yuvOrRgba[row1 + ix0] * (1 - dx) * dy +
          yuvOrRgba[row1 + ix1] * dx * dy;

        const uvRow = Math.floor(iy0 / 2);
        const uvCol = Math.floor(ix0 / 2);
        const uvIndex = frameW * frameH + uvRow * frameW + uvCol * 2;

        let uVal = 128;
        let vVal = 128;
        if (uvIndex + 1 < yuvOrRgba.length) {
          uVal = yuvOrRgba[uvIndex];
          vVal = yuvOrRgba[uvIndex + 1];
        }

        const u = uVal - 128;
        const v = vVal - 128;

        r = Y + 1.402 * v;
        g = Y - 0.344136 * u - 0.714136 * v;
        b = Y + 1.772 * u;

        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));

        sumR += r;
        sumB += b;
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

      const idx = oy * OUT + ox;
      tensor[bOff + idx] = b / 255.0;
      tensor[gOff + idx] = g / 255.0;
      tensor[rOff + idx] = r / 255.0;
    }
  }

  if (isYUV && sumB > sumR) {
    for (let idx = 0; idx < OUT * OUT; idx++) {
      const temp = tensor[bOff + idx];
      tensor[bOff + idx] = tensor[rOff + idx];
      tensor[rOff + idx] = temp;
    }
  }

  return tensor;
}

export async function runMiniFASNet(session, float32Input) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const inputTensor = new Tensor('float32', float32Input, [1, 3, 80, 80]);
  const result = await session.run({ [inputName]: inputTensor });
  return result[outputName].data;
}

export function getLivenessScore(logits) {
  const x0 = logits[0];
  const x1 = logits[1];
  const x2 = logits[2];

  const maxVal = Math.max(x0, x1, x2);
  const exp0 = Math.exp(x0 - maxVal);
  const exp1 = Math.exp(x1 - maxVal);
  const exp2 = Math.exp(x2 - maxVal);
  const sumExp = exp0 + exp1 + exp2;

  return exp2 / sumExp;
}
