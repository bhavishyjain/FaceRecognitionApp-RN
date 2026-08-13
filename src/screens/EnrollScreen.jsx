import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Platform,
  useWindowDimensions,
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
import Svg, { Path, Polyline, Ellipse } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import {
  preprocessFaceRGBA,
  getFaceEmbedding,
  l2normalize,
  EMBED_DIM,
} from '../helpers/faceRecognition';
import { saveEmbedding } from '../helpers/storage';

const ENROLL_STEPS = [
  { id: 'front', label: 'Look straight at the camera' },
  { id: 'left', label: 'Turn your head slightly left' },
  { id: 'right', label: 'Turn your head slightly right' },
  { id: 'up', label: 'Look slightly up' },
  { id: 'down', label: 'Look slightly down' },
];

const YAW_SIGN = 1;
const YAW_TARGET = 10;
const YAW_FRONT_TOL = 15;
const PITCH_TARGET = 6;
const PITCH_TOL = 20;

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function assessFaceQuality(faces, frameWidth, stepId) {
  if (faces.length === 0) {
    return { ok: false, reason: 'No face detected', progress: 0 };
  }
  if (faces.length > 1) {
    return { ok: false, reason: 'One face at a time', progress: 0 };
  }

  const face = faces[0];
  const ratio = face.bounds.width / frameWidth;
  if (ratio < 0.3) {
    return { ok: false, reason: 'Move closer', progress: 0 };
  }
  if (ratio > 0.85) {
    return { ok: false, reason: 'Move back', progress: 0 };
  }

  const yaw = (face.yawAngle ?? 0) * YAW_SIGN;
  const pitch = face.pitchAngle ?? 0;
  const yawR = Math.round(yaw);
  const pitchR = Math.round(pitch);
  let progress = 0;

  if (stepId === 'front') {
    progress = clamp01(
      1 - Math.max(Math.abs(yaw), Math.abs(pitch)) / YAW_FRONT_TOL,
    );
    if (Math.abs(yaw) > YAW_FRONT_TOL)
      return { ok: false, reason: `Look straight (yaw ${yawR}°)`, progress };
    if (Math.abs(pitch) > YAW_FRONT_TOL)
      return {
        ok: false,
        reason: `Level your head (pitch ${pitchR}°)`,
        progress,
      };
  } else if (stepId === 'left') {
    progress = clamp01(yaw / YAW_TARGET);
    if (yaw < YAW_TARGET)
      return {
        ok: false,
        reason: `Turn head left (yaw ${yawR}°/${YAW_TARGET}°)`,
        progress,
      };
    if (Math.abs(pitch) > PITCH_TOL)
      return {
        ok: false,
        reason: `Level your head (pitch ${pitchR}°)`,
        progress,
      };
  } else if (stepId === 'right') {
    progress = clamp01(-yaw / YAW_TARGET);
    if (yaw > -YAW_TARGET)
      return {
        ok: false,
        reason: `Turn head right (yaw ${yawR}°/-${YAW_TARGET}°)`,
        progress,
      };
    if (Math.abs(pitch) > PITCH_TOL)
      return {
        ok: false,
        reason: `Level your head (pitch ${pitchR}°)`,
        progress,
      };
  } else if (stepId === 'up') {
    progress = clamp01(pitch / PITCH_TARGET);
    if (Math.abs(yaw) > YAW_FRONT_TOL)
      return {
        ok: false,
        reason: `Face camera straight (yaw ${yawR}°)`,
        progress,
      };
    if (pitch < PITCH_TARGET)
      return {
        ok: false,
        reason: `Look slightly up (pitch ${pitchR}°/${PITCH_TARGET}°)`,
        progress,
      };
  } else if (stepId === 'down') {
    progress = clamp01(-pitch / PITCH_TARGET);
    if (Math.abs(yaw) > YAW_FRONT_TOL)
      return {
        ok: false,
        reason: `Face camera straight (yaw ${yawR}°)`,
        progress,
      };
    if (pitch > -PITCH_TARGET)
      return {
        ok: false,
        reason: `Look slightly down (pitch ${pitchR}°/-${PITCH_TARGET}°)`,
        progress,
      };
  }

  const le = face.leftEyeOpenProbability ?? 1;
  const re = face.rightEyeOpenProbability ?? 1;
  if (le < 0.5 || re < 0.5) {
    return { ok: false, reason: 'Open your eyes', progress };
  }

  return { ok: true, reason: 'Hold still…', progress: 1 };
}

const GOOD_MS = 900;
const COUNTDOWN = 3;

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

export default function EnrollScreen({ session, onDone }) {
  const insets = useSafeAreaInsets();
  const { width: screenW, height: screenH } = useWindowDimensions();
  const OVAL_W = 300;
  const OVAL_H = 360;
  const ovalCx = screenW / 2;
  const ovalCy = screenH / 2;
  const ovalRx = OVAL_W / 2;
  const ovalRy = OVAL_H / 2;
  const maskPath = `M0,0H${screenW}V${screenH}H0Z M${
    ovalCx - ovalRx
  },${ovalCy} A${ovalRx},${ovalRy} 0 1,0 ${
    ovalCx + ovalRx
  },${ovalCy} A${ovalRx},${ovalRy} 0 1,0 ${ovalCx - ovalRx},${ovalCy}Z`;
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
  ]);
  const cameraRef = useRef(null);

  const [hasPermission, setHasPermission] = useState(false);
  const [stage, setStage] = useState('scanning');
  const [quality, setQuality] = useState({
    ok: false,
    reason: 'Detecting…',
    progress: 0,
  });
  const [countdown, setCountdown] = useState(COUNTDOWN);
  const [name, setName] = useState('');
  const [pendingEmbed, setPendingEmbed] = useState(null);
  const [stepIndex, setStepIndex] = useState(0);
  const collectedEmbedsRef = useRef([]);
  const shouldCapture = useSharedValue(false);
  const currentFaceRef = useRef(null);
  const frameWidthRef = useRef(0);
  const frameHeightRef = useRef(0);

  const goodSinceRef = useRef(null);
  const capturedRef = useRef(false);
  const stepIndexRef = useRef(0);
  useEffect(() => {
    stepIndexRef.current = stepIndex;
  }, [stepIndex]);

  useEffect(() => {
    Camera.requestCameraPermission().then(s =>
      setHasPermission(s === 'granted'),
    );
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

  const processCapturedFrame = useCallback(
    async (cropArray, meanLuminance, isBlurry) => {
      if (!session) return;
      try {
        if (isBlurry) {
          throw new Error('Image is too blurry. Please hold still.');
        }
        if (meanLuminance < 40) {
          throw new Error('Image is too dark. Please improve lighting.');
        }
        if (meanLuminance > 220) {
          throw new Error('Image is too bright. Please reduce lighting.');
        }

        setStage('processing');

        const float32 = new Float32Array(cropArray);
        const embedding = await getFaceEmbedding(float32, session);

        collectedEmbedsRef.current.push(embedding);

        const nextStep = stepIndexRef.current + 1;
        if (nextStep < 5) {
          setStepIndex(nextStep);
          capturedRef.current = false;
          goodSinceRef.current = null;
          setStage('scanning');
        } else {
          const sum = new Float32Array(EMBED_DIM);
          for (const emb of collectedEmbedsRef.current) {
            for (let i = 0; i < EMBED_DIM; i++) {
              sum[i] += emb[i];
            }
          }
          const avgEmbed = l2normalize(sum);
          setPendingEmbed(avgEmbed);
          setStage('naming');
        }
      } catch (err) {
        console.error('[Enroll] processing error:', err);
        capturedRef.current = false;
        goodSinceRef.current = null;
        setStage('scanning');
        Alert.alert('Processing failed', err.message || String(err));
      }
    },
    [session],
  );

  const onFacesDetectedJS = useCallback(
    (faces, frameW, frameH, cropArray, meanLuminance, isBlurry) => {
      frameWidthRef.current = frameW;
      frameHeightRef.current = frameH;
      currentFaceRef.current = faces.length === 1 ? faces[0] : null;

      if (cropArray) {
        processCapturedFrame(cropArray, meanLuminance, isBlurry);
        return;
      }

      const currentStep = ENROLL_STEPS[stepIndexRef.current]?.id || 'front';
      const q = assessFaceQuality(faces, frameW, currentStep);
      setQuality(q);

      if (capturedRef.current) {
        return;
      }

      if (q.ok) {
        if (!goodSinceRef.current) {
          goodSinceRef.current = Date.now();
        }
        const elapsed = Date.now() - goodSinceRef.current;
        const remaining = Math.ceil(
          (GOOD_MS - elapsed) / (GOOD_MS / COUNTDOWN),
        );
        setCountdown(Math.max(1, remaining));
        setStage('countdown');
        if (elapsed >= GOOD_MS) {
          capturedRef.current = true;
          shouldCapture.value = true;
        }
      } else {
        goodSinceRef.current = null;
        setCountdown(COUNTDOWN);
        setStage('scanning');
      }
    },
    [processCapturedFrame, shouldCapture],
  );

  const onFacesDetected = useMemo(
    () => Worklets.createRunOnJS(onFacesDetectedJS),
    [onFacesDetectedJS],
  );

  const frameSkip = useSharedValue(0);
  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      frameSkip.value = (frameSkip.value + 1) % 3;
      if (frameSkip.value !== 0) return;
      runAsync(frame, () => {
        'worklet';
        const faces = detectFaces(frame);
        const primaryFace = getPrimaryFace(faces);
        const selectedFaces = primaryFace ? [primaryFace] : [];
        if (shouldCapture.value) {
          shouldCapture.value = false;
          let cropArray = null;
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
          }
          onFacesDetected(
            selectedFaces,
            frame.width,
            frame.height,
            cropArray,
            meanLuminance,
            isBlurry,
          );
        } else {
          onFacesDetected(
            selectedFaces,
            frame.width,
            frame.height,
            null,
            0,
            false,
          );
        }
      });
    },
    [onFacesDetected, preprocessFaceRGBA, getPrimaryFace],
  );

  const handleSave = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      Alert.alert('Enter a name');
      return;
    }
    if (!pendingEmbed) {
      return;
    }
    setStage('saving');
    try {
      await saveEmbedding(trimmed, pendingEmbed);
      Alert.alert('Enrolled!', `"${trimmed}" saved.`, [
        { text: 'OK', onPress: onDone },
      ]);
    } catch (e) {
      Alert.alert('Save failed', String(e));
      setStage('naming');
    }
  }, [name, pendingEmbed, onDone]);

  const handleRetry = () => {
    capturedRef.current = false;
    goodSinceRef.current = null;
    currentFaceRef.current = null;
    frameWidthRef.current = 0;
    frameHeightRef.current = 0;
    collectedEmbedsRef.current = [];
    setStepIndex(0);
    setPendingEmbed(null);
    setName('');
    setCountdown(COUNTDOWN);
    setStage('scanning');
  };

  const isCameraActive =
    stage === 'scanning' || stage === 'countdown' || stage === 'processing';

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

  if (stage === 'naming' || stage === 'saving') {
    return (
      <View style={s.namingContainer}>
        <View style={s.successIconRing}>
          <Svg
            width={32}
            height={32}
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
        </View>

        <Text style={s.namingTitle}>Face Captured</Text>
        <Text style={s.namingSubtitle}>Give this face a name to save it</Text>

        <View style={s.inputCard}>
          <Text style={s.inputLabel}>NAME</Text>
          <TextInput
            id="input-enroll-name"
            style={s.nameInput}
            placeholder="Enter name"
            placeholderTextColor={C.textTertiary}
            value={name}
            onChangeText={setName}
            autoFocus
            returnKeyType="done"
            onSubmitEditing={handleSave}
          />
        </View>

        <View style={s.namingBtns}>
          <TouchableOpacity
            id="btn-save-enrollment"
            style={[s.namingBtn, s.btnSave]}
            onPress={handleSave}
            disabled={stage === 'saving'}
            activeOpacity={0.85}
          >
            {stage === 'saving' ? (
              <ActivityIndicator color={C.surface} />
            ) : (
              <>
                <Svg
                  width={18}
                  height={18}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={C.surface}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <Path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                  <Polyline points="17 21 17 13 7 13 7 21" />
                  <Polyline points="7 3 7 8 15 8" />
                </Svg>
                <Text style={s.saveBtnText}>Save</Text>
              </>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            id="btn-retake"
            style={[s.namingBtn, s.btnRetry]}
            onPress={handleRetry}
            activeOpacity={0.85}
          >
            <Svg
              width={18}
              height={18}
              viewBox="0 0 24 24"
              fill="none"
              stroke={C.text}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <Path d="M21 2v6h-6" />
              <Path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
              <Path d="M3 22v-6h6" />
              <Path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
            </Svg>
            <Text style={s.retakeBtnText}>Retake</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const isGood = quality.ok && stage === 'countdown';

  const statusLabel =
    stage === 'processing'
      ? 'Processing face…'
      : ENROLL_STEPS[stepIndex]
      ? `Pose ${stepIndex + 1}/5: ${ENROLL_STEPS[stepIndex].label}`
      : quality.reason;

  const statusSub =
    stage === 'processing'
      ? 'Generating face embedding'
      : stage === 'countdown'
      ? `Capturing in ${countdown}…`
      : quality.reason;

  return (
    <View style={s.container}>
      <Camera
        ref={cameraRef}
        style={[StyleSheet.absoluteFill, { transform: [{ scaleX: -1 }] }]}
        device={device}
        format={format}
        isActive={isCameraActive}
        frameProcessor={frameProcessor}
        pixelFormat={Platform.OS === 'ios' ? 'rgb' : 'yuv'}
      />

      <Svg
        pointerEvents="none"
        style={StyleSheet.absoluteFill}
        width={screenW}
        height={screenH}
      >
        <Path d={maskPath} fill="rgba(0,0,0,0.65)" fillRule="evenodd" />
        <Ellipse
          cx={ovalCx}
          cy={ovalCy}
          rx={ovalRx}
          ry={ovalRy}
          fill="none"
          stroke={isGood ? C.success : 'rgba(250,250,250,0.6)'}
          strokeWidth={isGood ? 3 : 2}
        />
      </Svg>

      <View style={[s.topBar, { paddingTop: insets.top + 12 }]}>
        <TouchableOpacity
          id="btn-back-enroll"
          style={s.topBarBtn}
          onPress={onDone}
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
        <Text style={s.topTitle}>Enroll Face</Text>
        <View style={s.topBarBtn} />
      </View>

      <View style={[s.instructionCard, { top: insets.top + 76 }]}>
        <View style={[s.statusIconContainer, isGood && s.statusIconGood]}>
          {stage === 'processing' ? (
            <ActivityIndicator color={C.accent} size="small" />
          ) : (
            <View style={[s.dot, isGood ? s.dotGreen : s.dotRed]} />
          )}
        </View>

        <View style={s.statusTextGroup}>
          <Text style={[s.statusLabel, isGood && { color: C.success }]}>
            {statusLabel}
          </Text>
          <Text style={s.statusSub}>{statusSub}</Text>
          {stage !== 'processing' && (
            <View style={s.progressTrack}>
              <View
                style={[
                  s.progressFill,
                  {
                    width: `${Math.round(quality.progress * 100)}%`,
                    backgroundColor: isGood ? C.success : C.accent,
                  },
                ]}
              />
            </View>
          )}
        </View>

        {stage === 'countdown' && (
          <Text style={s.countdownBig}>{countdown}</Text>
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
  topBarBtn: {
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
    textAlignVertical: 'center',
  },

  instructionCard: {
    position: 'absolute',
    left: 16,
    right: 16,
    backgroundColor: 'rgba(9,9,11,0.82)',
    borderRadius: 20,
    paddingVertical: 14,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(250,250,250,0.1)',
    gap: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  statusIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.card,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusIconGood: {
    backgroundColor: C.success + '18',
  },
  dot: { width: 12, height: 12, borderRadius: 6 },
  dotGreen: { backgroundColor: C.success },
  dotRed: { backgroundColor: C.danger },
  statusTextGroup: { flex: 1 },
  statusLabel: {
    color: C.text,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '700',
    textAlignVertical: 'center',
  },
  statusSub: {
    color: C.textSecondary,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 2,
    textAlignVertical: 'center',
  },
  progressTrack: {
    height: 4,
    borderRadius: 2,
    backgroundColor: C.cardElevated,
    marginTop: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  countdownBig: {
    color: C.success,
    fontSize: 32,
    lineHeight: 40,
    fontWeight: '800',
  },

  namingContainer: {
    flex: 1,
    backgroundColor: C.surface,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  successIconRing: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: C.success + '15',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  namingTitle: {
    color: C.text,
    fontSize: 24,
    lineHeight: 32,
    fontWeight: '800',
    marginBottom: 6,
    textAlignVertical: 'center',
  },
  namingSubtitle: {
    color: C.textSecondary,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 28,
    textAlignVertical: 'center',
  },
  inputCard: {
    width: '100%',
    backgroundColor: C.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: C.border,
  },
  inputLabel: {
    color: C.textTertiary,
    fontSize: 11,
    lineHeight: 16,
    fontWeight: '700',
    letterSpacing: 1.2,
    marginBottom: 8,
    textAlignVertical: 'center',
  },
  nameInput: {
    width: '100%',
    backgroundColor: C.surface,
    color: C.text,
    fontSize: 18,
    lineHeight: 26,
    fontWeight: '600',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: C.border,
  },
  namingBtns: { flexDirection: 'row', gap: 12, width: '100%' },
  namingBtn: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  btnSave: { backgroundColor: C.primary },
  btnRetry: { backgroundColor: C.cardElevated },
  saveBtnText: {
    color: C.surface,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '700',
    textAlignVertical: 'center',
  },
  retakeBtnText: {
    color: C.text,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '700',
    textAlignVertical: 'center',
  },
});
