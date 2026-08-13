import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Platform,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useFrameProcessor,
  runAsync,
} from 'react-native-vision-camera';
import { useFaceDetector } from 'react-native-vision-camera-face-detector';
import { Worklets, useSharedValue } from 'react-native-worklets-core';
import Svg, { Path, Circle, Polyline } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  preprocessFaceRGBA,
  preprocessFaceFAS,
  getFaceEmbedding,
  runMiniFASNet,
  getLivenessScore,
  LIVENESS_THRESHOLD,
  cosineSimilarity,
  MATCH_THRESHOLD,
} from '../helpers/faceRecognition';
import { loadAllEmbeddings } from '../helpers/storage';

const INFERENCE_INTERVAL_MS = 150;
const TRACKING_THRESHOLD = 3;

function isSameFace(face1, face2, frameWidth, frameHeight) {
  if (!face1 || !face2) return false;

  if (
    face1.trackingId !== undefined &&
    face1.trackingId !== null &&
    face2.trackingId !== undefined &&
    face2.trackingId !== null
  ) {
    return face1.trackingId === face2.trackingId;
  }

  const b1 = face1.bounds;
  const b2 = face2.bounds;
  if (!b1 || !b2) return false;

  const c1x = b1.x + b1.width / 2;
  const c1y = b1.y + b1.height / 2;
  const c2x = b2.x + b2.width / 2;
  const c2y = b2.y + b2.height / 2;

  const dist = Math.sqrt(Math.pow(c1x - c2x, 2) + Math.pow(c1y - c2y, 2));
  const maxDim = Math.max(frameWidth, frameHeight) || 1;
  const normalizedDist = dist / maxDim;

  const area1 = b1.width * b1.height;
  const area2 = b2.width * b2.height;
  const sizeRatio = Math.max(area1, area2) / Math.min(area1, area2);

  return normalizedDist < 0.15 && sizeRatio < 1.5;
}

function assessFaceQuality(faces, frameWidth) {
  if (faces.length === 0) {
    return { ok: false, reason: 'No face detected' };
  }
  if (faces.length > 1) {
    return { ok: false, reason: 'One face at a time' };
  }
  const face = faces[0];
  const ratio = face.bounds.width / frameWidth;
  if (ratio < 0.25) {
    return { ok: false, reason: 'Move closer' };
  }
  if (ratio > 0.5) {
    return { ok: false, reason: 'Move back' };
  }
  if (Math.abs(face.yawAngle ?? 0) > 20) {
    return { ok: false, reason: 'Face camera straight on' };
  }
  if (Math.abs(face.pitchAngle ?? 0) > 20) {
    return { ok: false, reason: 'Level your head' };
  }
  return { ok: true, reason: 'Analysing…' };
}

function getPrimaryFace(faces) {
  'worklet';
  if (!faces || faces.length === 0) {
    return null;
  }
  let primary = faces[0];
  let maxArea = primary.bounds.width * primary.bounds.height;
  for (let i = 1; i < faces.length; i++) {
    const area = faces[i].bounds.width * faces[i].bounds.height;
    if (area > maxArea) {
      primary = faces[i];
      maxArea = area;
    }
  }
  return primary;
}

export default function VerifyScreen({ session, fasSession, onBack }) {
  const insets = useSafeAreaInsets();
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
  ]);

  const [hasPermission, setHasPermission] = useState(false);
  const [quality, setQuality] = useState({ ok: false, reason: 'Detecting…' });
  const [match, setMatch] = useState(null);
  const [isInferring, setIsInferring] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [enrolled, setEnrolled] = useState([]);
  const shouldCapture = useSharedValue(false);
  const lastInferenceRef = useRef(0);
  const inferringRef = useRef(false);
  const verifiedRef = useRef(false);
  const trackedFramesCountRef = useRef(0);
  const prevFaceRef = useRef(null);
  const blinkPhaseRef = useRef('eyes-open');
  const [blinkDetected, setBlinkDetected] = useState(false);

  useEffect(() => {
    Camera.requestCameraPermission().then(s =>
      setHasPermission(s === 'granted'),
    );
    loadAllEmbeddings().then(setEnrolled);
  }, []);

  const faceOptions = useRef({
    performanceMode: 'fast',
    landmarkMode: 'all',
    contourMode: 'none',
    classificationMode: 'all',
    minFaceSize: 0.15,
  }).current;
  const { detectFaces, stopListeners } = useFaceDetector(faceOptions);
  useEffect(() => () => stopListeners(), [stopListeners]);

  const runInference = useCallback(
    async (cropArray, fasCropArray, meanLuminance, isBlurry) => {
      if (
        !session ||
        !fasSession ||
        inferringRef.current ||
        verifiedRef.current
      ) {
        return;
      }
      inferringRef.current = true;
      setIsInferring(true);

      try {
        if (isBlurry) {
          setQuality({ ok: false, reason: 'Blurry — hold still' });
          return;
        }
        if (meanLuminance < 40) {
          setQuality({ ok: false, reason: 'Too dark' });
          return;
        }
        if (meanLuminance > 220) {
          setQuality({ ok: false, reason: 'Too bright' });
          return;
        }

        const fasFloat32 = new Float32Array(fasCropArray);
        const livenessLogits = await runMiniFASNet(fasSession, fasFloat32);
        const livenessScore = getLivenessScore(livenessLogits);
        const livenessPassed = livenessScore >= LIVENESS_THRESHOLD;

        if (!livenessPassed) {
          setQuality({ ok: false, reason: 'Spoof detected' });
          setMatch({
            name: 'Spoof Detected',
            similarity: 0,
            matched: false,
            isSpoof: true,
          });
          return;
        }

        const float32 = new Float32Array(cropArray);
        const embed = await getFaceEmbedding(float32, session);

        let bestName = '';
        let bestSim = -1;
        for (const ef of enrolled) {
          const sim = cosineSimilarity(embed, ef.embedding);
          if (sim > bestSim) {
            bestSim = sim;
            bestName = ef.name;
          }
        }

        const matched = bestSim >= MATCH_THRESHOLD;
        setMatch({
          name: bestName,
          similarity: bestSim,
          matched,
          isSpoof: false,
        });
        if (matched) {
          verifiedRef.current = true;
          setIsVerified(true);
          setQuality({ ok: true, reason: 'Verified' });
        } else {
          setQuality({ ok: false, reason: 'Not recognized' });
        }
      } catch (err) {
        console.error('[Verify] inference error:', err);
      } finally {
        inferringRef.current = false;
        setIsInferring(false);
      }
    },
    [session, fasSession, enrolled],
  );

  const onFacesDetectedJS = useCallback(
    (faces, fw, fh, cropArray, fasCropArray, meanLuminance, isBlurry) => {
      if (verifiedRef.current) {
        return;
      }

      if (cropArray && fasCropArray) {
        runInference(cropArray, fasCropArray, meanLuminance, isBlurry);
        return;
      }

      if (faces.length === 0) {
        trackedFramesCountRef.current = 0;
        prevFaceRef.current = null;
        blinkPhaseRef.current = 'eyes-open';
        setBlinkDetected(false);
        setQuality({ ok: false, reason: 'No face detected' });
        setMatch(null);
        return;
      }

      const currentFace = faces[0];
      const isSame = isSameFace(currentFace, prevFaceRef.current, fw, fh);

      if (isSame) {
        trackedFramesCountRef.current += 1;
      } else {
        trackedFramesCountRef.current = 1;
        blinkPhaseRef.current = 'eyes-open';
        setBlinkDetected(false);
        setMatch(null);
      }
      prevFaceRef.current = currentFace;

      const count = Math.min(trackedFramesCountRef.current, TRACKING_THRESHOLD);

      if (trackedFramesCountRef.current < TRACKING_THRESHOLD) {
        setQuality({
          ok: false,
          reason: `Stabilizing… (${count}/${TRACKING_THRESHOLD})`,
        });
        return;
      }

      const q = assessFaceQuality(faces, fw);
      if (!q.ok) {
        setQuality(q);
        return;
      }

      if (blinkPhaseRef.current !== 'done') {
        const le = currentFace.leftEyeOpenProbability ?? 1;
        const re = currentFace.rightEyeOpenProbability ?? 1;
        const bothOpen = le > 0.75 && re > 0.75;
        const bothClosed = le < 0.35 && re < 0.35;

        if (blinkPhaseRef.current === 'eyes-open' && bothClosed) {
          blinkPhaseRef.current = 'eyes-closed';
        } else if (blinkPhaseRef.current === 'eyes-closed' && bothOpen) {
          blinkPhaseRef.current = 'done';
          setBlinkDetected(true);
          lastInferenceRef.current = Date.now();
          shouldCapture.value = true;
        }

        setQuality({ ok: true, reason: 'Blink once to verify' });
        return;
      }

      setQuality({ ok: true, reason: 'Hold still…' });
      const now = Date.now();
      if (
        !inferringRef.current &&
        now - lastInferenceRef.current >= INFERENCE_INTERVAL_MS
      ) {
        lastInferenceRef.current = now;
        shouldCapture.value = true;
      }
    },
    [runInference, shouldCapture],
  );

  const onFacesDetected = useMemo(
    () => Worklets.createRunOnJS(onFacesDetectedJS),
    [onFacesDetectedJS],
  );

  const frameSkip = useSharedValue(0);
  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      frameSkip.value = (frameSkip.value + 1) % 5;
      if (frameSkip.value !== 0) {
        return;
      }
      runAsync(frame, () => {
        'worklet';
        const faces = detectFaces(frame);
        const primaryFace = getPrimaryFace(faces);
        const selectedFaces = primaryFace ? [primaryFace] : [];
        if (shouldCapture.value) {
          shouldCapture.value = false;
          let cropArray = null;
          let fasCropArray = null;
          let meanLuminance = 0;
          let isBlurry = false;
          if (primaryFace) {
            const buffer = frame.toArrayBuffer();
            const yuv = new Uint8Array(buffer);

            const float32 = preprocessFaceRGBA(
              yuv,
              frame.width,
              frame.height,
              primaryFace,
            );
            cropArray = Array.from(float32);
            meanLuminance = float32.meanLuminance;
            isBlurry = float32.isBlurry;

            const fasFloat32 = preprocessFaceFAS(
              yuv,
              frame.width,
              frame.height,
              primaryFace,
              2.7,
            );
            fasCropArray = Array.from(fasFloat32);
          }
          onFacesDetected(
            selectedFaces,
            frame.width,
            frame.height,
            cropArray,
            fasCropArray,
            meanLuminance,
            isBlurry,
          );
        } else {
          onFacesDetected(
            selectedFaces,
            frame.width,
            frame.height,
            null,
            null,
            0,
            false,
          );
        }
      });
    },
    [onFacesDetected, preprocessFaceRGBA, preprocessFaceFAS, getPrimaryFace],
  );

  if (!hasPermission) {
    return (
      <View style={s.center}>
        <Text style={s.white}>Camera permission required.</Text>
      </View>
    );
  }
  if (!device) {
    return (
      <View style={s.center}>
        <Text style={s.white}>No front camera found.</Text>
      </View>
    );
  }

  const isSpoof = match?.isSpoof;
  const isMatched = match?.matched;
  const isNoMatch = match && !match.isSpoof && !match.matched && !isInferring;

  const statusColor = isMatched
    ? C.success
    : isSpoof || isNoMatch
    ? C.danger
    : C.text;

  const matchLabel = isInferring
    ? 'Scanning…'
    : isSpoof
    ? 'Spoof Detected'
    : isMatched
    ? match.name
    : isNoMatch
    ? 'Not Recognized'
    : blinkDetected
    ? 'Verifying…'
    : 'Ready to Verify';

  const matchSub = isInferring
    ? 'Running face recognition'
    : isSpoof
    ? 'Liveness check failed'
    : isMatched
    ? `${(match.similarity * 100).toFixed(1)}% match`
    : isNoMatch
    ? `${(match.similarity * 100).toFixed(1)}% — below threshold`
    : blinkDetected
    ? 'Processing…'
    : 'Look at the camera, then blink once';

  return (
    <View style={s.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={!isVerified}
        frameProcessor={isVerified ? undefined : frameProcessor}
        pixelFormat={Platform.OS === 'ios' ? 'rgb' : 'yuv'}
      />

      <View style={[s.topBar, { paddingTop: insets.top + 10 }]}>
        <TouchableOpacity
          id="btn-back-verify"
          style={s.backBtn}
          onPress={onBack}
          activeOpacity={0.7}
        >
          <Svg
            width={18}
            height={18}
            viewBox="0 0 24 24"
            fill="none"
            stroke="#fff"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <Path d="M19 12H5" />
            <Path d="M12 19l-7-7 7-7" />
          </Svg>
        </TouchableOpacity>
        <Text style={s.topTitle}>Verify Face</Text>
        <View style={s.backBtn} />
      </View>

      <View style={s.statusBadge}>
        <View
          style={[
            s.statusDot,
            { backgroundColor: quality.ok ? C.success : C.amber },
          ]}
        />
        <Text style={s.statusBadgeText}>{quality.reason}</Text>
      </View>

      <View style={[s.panel, { paddingBottom: insets.bottom + 20 }]}>
        <View style={s.panelInner}>
          <View style={[s.iconRing, { borderColor: statusColor + '40' }]}>
            {isInferring ? (
              <ActivityIndicator color={statusColor} size="small" />
            ) : isSpoof ? (
              <Svg
                width={24}
                height={24}
                viewBox="0 0 24 24"
                fill="none"
                stroke={C.danger}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <Circle cx="12" cy="12" r="10" />
                <Path d="M15 9l-6 6" />
                <Path d="M9 9l6 6" />
              </Svg>
            ) : isMatched ? (
              <Svg
                width={24}
                height={24}
                viewBox="0 0 24 24"
                fill="none"
                stroke={C.success}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <Path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <Polyline points="22 4 12 14.01 9 11.01" />
              </Svg>
            ) : isNoMatch ? (
              <Svg
                width={24}
                height={24}
                viewBox="0 0 24 24"
                fill="none"
                stroke={C.danger}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <Circle cx="12" cy="12" r="10" />
                <Path d="M15 9l-6 6" />
                <Path d="M9 9l6 6" />
              </Svg>
            ) : (
              <Svg
                width={24}
                height={24}
                viewBox="0 0 24 24"
                fill="none"
                stroke={C.muted}
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <Path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <Circle cx="12" cy="12" r="3" />
              </Svg>
            )}
          </View>

          <View style={s.textBlock}>
            <Text style={[s.label, { color: statusColor }]}>{matchLabel}</Text>
            <Text style={s.sub}>{matchSub}</Text>
          </View>
        </View>

        {match && !isSpoof && !isInferring && (
          <View style={s.simTrack}>
            <View
              style={[
                s.simFill,
                {
                  width: `${Math.max(match.similarity * 100, 2)}%`,
                  backgroundColor: isMatched ? C.success : C.danger,
                },
              ]}
            />
          </View>
        )}

        {isVerified && (
          <TouchableOpacity
            style={s.doneBtn}
            onPress={onBack}
            activeOpacity={0.8}
          >
            <Text style={s.doneBtnText}>Done</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const C = {
  bg: '#0A0A0A',
  panel: '#111111',
  border: '#1F1F1F',
  text: '#F5F5F5',
  muted: '#555555',
  sub: '#888888',
  accent: '#3B82F6',
  success: '#22C55E',
  danger: '#EF4444',
  amber: '#F59E0B',
};

const s = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  center: {
    flex: 1,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  white: {
    color: C.text,
    fontSize: 15,
  },

  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(10,10,10,0.72)',
  },
  backBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  topTitle: {
    color: C.text,
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 0.2,
    lineHeight: 20,
  },

  statusBadge: {
    position: 'absolute',
    top: 110,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(10,10,10,0.75)',
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    gap: 7,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
  statusBadgeText: {
    lineHeight: 16,
    color: C.text,
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 0.1,
  },

  panel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: C.panel,
    borderTopWidth: 1,
    borderTopColor: C.border,
    paddingTop: 24,
    paddingHorizontal: 20,
  },
  panelInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 18,
  },
  iconRing: {
    width: 48,
    height: 48,
    borderRadius: 12,
    borderWidth: 1.5,
    backgroundColor: 'rgba(255,255,255,0.04)',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  textBlock: {
    flex: 1,
  },
  label: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 3,
    letterSpacing: -0.2,
    lineHeight: 22,
  },
  sub: {
    color: C.sub,
    fontSize: 13,
    lineHeight: 24,
  },

  simTrack: {
    height: 2,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 1,
    overflow: 'hidden',
    marginBottom: 20,
  },
  simFill: {
    height: '100%',
    borderRadius: 1,
  },

  doneBtn: {
    backgroundColor: C.text,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    marginBottom: 4,
  },
  doneBtnText: {
    color: C.bg,
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 0.1,
  },
});
