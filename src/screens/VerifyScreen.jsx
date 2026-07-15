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
  getFaceEmbedding,
  cosineSimilarity,
  MATCH_THRESHOLD,
} from '../helpers/faceRecognition';
import { loadAllEmbeddings } from '../helpers/storage';

function assessFaceQuality(faces, frameWidth) {
  if (faces.length === 0) {
    return { ok: false, reason: 'No face detected' };
  }
  if (faces.length > 1) {
    return { ok: false, reason: 'One face at a time' };
  }
  const face = faces[0];
  if (face.bounds.width / frameWidth < 0.25) {
    return { ok: false, reason: 'Move closer' };
  }
  if (Math.abs(face.yawAngle ?? 0) > 20) {
    return { ok: false, reason: 'Face camera straight on' };
  }
  if (Math.abs(face.pitchAngle ?? 0) > 20) {
    return { ok: false, reason: 'Level your head' };
  }
  return { ok: true, reason: 'Analysing…' };
}

const INFERENCE_INTERVAL_MS = 1000;

export default function VerifyScreen({ session, onBack }) {
  const insets = useSafeAreaInsets();
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
  ]);
  const cameraRef = useRef(null);

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
    classificationMode: 'none',
    minFaceSize: 0.15,
  }).current;
  const { detectFaces, stopListeners } = useFaceDetector(faceOptions);
  useEffect(() => () => stopListeners(), [stopListeners]);

  const runInference = useCallback(
    async (cropArray, meanLuminance, isBlurry) => {
      if (!session || inferringRef.current || verifiedRef.current) {
        return;
      }
      inferringRef.current = true;
      setIsInferring(true);

      try {
        if (isBlurry) {
          setQuality({ ok: false, reason: 'Blurry - hold still' });
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

        if (bestName) {
          console.log(
            `[Verify] Similarity score for ${bestName}: ${bestSim.toFixed(4)}`,
          );
        }

        const matched = bestSim >= MATCH_THRESHOLD;
        setMatch({ name: bestName, similarity: bestSim, matched });
        if (matched) {
          verifiedRef.current = true;
          setIsVerified(true);
          setQuality({ ok: true, reason: 'Verified' });
        }
      } catch (err) {
        console.error('[Verify] inference error:', err);
      } finally {
        inferringRef.current = false;
        setIsInferring(false);
      }
    },
    [session, enrolled],
  );

  const onFacesDetectedJS = useCallback(
    (faces, fw, fh, cropArray, meanLuminance, isBlurry) => {
      if (verifiedRef.current) {
        return;
      }

      if (cropArray) {
        runInference(cropArray, meanLuminance, isBlurry);
        return;
      }

      const q = assessFaceQuality(faces, fw);
      setQuality(q);

      const now = Date.now();
      if (
        q.ok &&
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
        if (shouldCapture.value) {
          shouldCapture.value = false;
          let cropArray = null;
          let meanLuminance = 0;
          let isBlurry = false;
          if (faces.length === 1) {
            const buffer = frame.toArrayBuffer();
            const yuv = new Uint8Array(buffer);
            const float32 = preprocessFaceRGBA(
              yuv,
              frame.width,
              frame.height,
              faces[0],
            );
            cropArray = Array.from(float32);
            meanLuminance = float32.meanLuminance;
            isBlurry = float32.isBlurry;
          }
          onFacesDetected(
            faces,
            frame.width,
            frame.height,
            cropArray,
            meanLuminance,
            isBlurry,
          );
        } else {
          onFacesDetected(faces, frame.width, frame.height, null, 0, false);
        }
      });
    },
    [onFacesDetected, preprocessFaceRGBA],
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

  const matchLabel = isInferring
    ? 'Scanning…'
    : match?.matched
    ? match.name
    : match
    ? 'No match'
    : 'Position your face in view';

  const matchSim =
    match && !isInferring
      ? `${(match.similarity * 100).toFixed(1)}% confidence`
      : isInferring
      ? 'Comparing against enrolled faces'
      : 'The entire screen is the capture area';

  return (
    <View style={s.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={!isVerified}
        frameProcessor={isVerified ? undefined : frameProcessor}
        pixelFormat={Platform.OS === 'ios' ? 'rgb' : 'yuv'}
      />

      <View style={[s.topBar, { paddingTop: insets.top + 12 }]}>
        <TouchableOpacity
          id="btn-back-verify"
          style={s.backBtn}
          onPress={onBack}
        >
          <Svg
            width={20}
            height={20}
            viewBox="0 0 24 24"
            fill="none"
            stroke="#fff"
            strokeWidth={2}
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

      <View style={s.qualityPill}>
        <View style={[s.dot, quality.ok ? s.dotGreen : s.dotRed]} />
        <Text style={s.qualityText}>{quality.reason}</Text>
      </View>

      <View style={[s.bottomCard, { paddingBottom: insets.bottom + 16 }]}>
        <View
          style={[
            s.statusIconContainer,
            match?.matched && s.statusIconSuccess,
            match && !match.matched && !isInferring && s.statusIconDanger,
          ]}
        >
          {isInferring ? (
            <ActivityIndicator color={C.accent} size="small" />
          ) : match?.matched ? (
            <Svg
              width={28}
              height={28}
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
          ) : match ? (
            <Svg
              width={28}
              height={28}
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
              width={28}
              height={28}
              viewBox="0 0 24 24"
              fill="none"
              stroke={C.primaryMuted}
              strokeWidth={1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <Path d="M4 8V6a2 2 0 0 1 2-2h2" />
              <Path d="M4 16v2a2 2 0 0 1 2 2h2" />
              <Path d="M16 4h2a2 2 0 0 1 2 2v2" />
              <Path d="M16 20h2a2 2 0 0 1 2-2v-2" />
            </Svg>
          )}
        </View>

        <Text
          style={[
            s.resultName,
            match?.matched && { color: C.success },
            match && !match.matched && !isInferring && { color: C.danger },
          ]}
        >
          {matchLabel}
        </Text>
        <Text style={s.resultSub}>{matchSim}</Text>

        {match && !isInferring && (
          <View style={s.simBarTrack}>
            <View
              style={[
                s.simBarFill,
                {
                  width: `${Math.max(match.similarity * 100, 2)}%`,
                  backgroundColor: match.matched ? C.success : C.danger,
                },
              ]}
            />
          </View>
        )}
        {isVerified && (
          <TouchableOpacity
            style={s.doneBtn}
            onPress={onBack}
            activeOpacity={0.85}
          >
            <Text style={s.doneBtnText}>Done</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const C = {
  surface: '#09090B',
  card: '#18181B',
  cardElevated: '#27272A',
  border: '#27272A',
  primary: '#FAFAFA',
  primaryMuted: '#A1A1AA',
  accent: '#3B82F6',
  success: '#22C55E',
  danger: '#EF4444',
  text: '#FAFAFA',
  textSecondary: '#A1A1AA',
  textTertiary: '#52525B',
};

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    backgroundColor: C.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  white: { color: C.text, fontSize: 16, lineHeight: 22 },

  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 12,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(9,9,11,0.5)',
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  topTitle: {
    color: C.text,
    fontSize: 17,
    lineHeight: 24,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  qualityPill: {
    position: 'absolute',
    top: 108,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(9,9,11,0.6)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 6,
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  dotGreen: { backgroundColor: C.success },
  dotRed: { backgroundColor: C.danger },
  qualityText: {
    color: C.text,
    fontSize: 12,
    lineHeight: 18,
    fontWeight: '600',
  },

  bottomCard: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: C.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 28,
    paddingBottom: 16,
    paddingHorizontal: 24,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  statusIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: C.card,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  statusIconSuccess: {
    backgroundColor: C.success + '18',
  },
  statusIconDanger: {
    backgroundColor: C.danger + '18',
  },
  resultName: {
    color: C.text,
    fontSize: 22,
    lineHeight: 30,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 4,
  },
  resultSub: {
    color: C.textSecondary,
    fontSize: 13,
    lineHeight: 18,
    textAlign: 'center',
    marginBottom: 16,
  },

  simBarTrack: {
    width: '100%',
    height: 4,
    backgroundColor: C.card,
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 8,
  },
  simBarFill: {
    height: '100%',
    borderRadius: 2,
  },
  doneBtn: {
    marginTop: 12,
    width: '100%',
    backgroundColor: C.primary,
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  doneBtnText: {
    color: C.surface,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '700',
  },
});
