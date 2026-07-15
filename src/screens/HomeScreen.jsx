import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import Svg, { Path, Circle, Line } from 'react-native-svg';
import { deleteEmbedding, loadEnrolledNames } from '../helpers/storage';

export default function HomeScreen({ onEnroll, onVerify }) {
  const [names, setNames] = useState([]);

  const refresh = useCallback(async () => {
    setNames(await loadEnrolledNames());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDelete = name =>
    Alert.alert('Remove face', `Delete "${name}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteEmbedding(name);
          refresh();
        },
      },
    ]);

  return (
    <View style={s.container}>
      <View style={s.header}>
        <Svg
          width={52}
          height={52}
          viewBox="0 0 24 24"
          fill="none"
          stroke={C.primary}
          strokeWidth={1.4}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <Path d="M4 8V6a2 2 0 0 1 2-2h2" />
          <Path d="M20 8V6a2 2 0 0 0-2-2h-2" />
          <Path d="M4 16v2a2 2 0 0 0 2 2h2" />
          <Path d="M20 16v2a2 2 0 0 1-2 2h-2" />

          <Circle cx="12" cy="12" r="4.5" />
          <Circle cx="10.2" cy="11" r="0.5" fill={C.primary} stroke="none" />
          <Circle cx="13.8" cy="11" r="0.5" fill={C.primary} stroke="none" />
          <Path d="M10.5 13c.5.7 1 1 1.5 1s1-.3 1.5-1" strokeWidth={1} />
        </Svg>

        <Text style={s.title}>Face Recognition</Text>
        <Text style={s.subtitle}>
          {names.length === 0
            ? 'No faces enrolled yet'
            : `${names.length} face${names.length > 1 ? 's' : ''} enrolled`}
        </Text>
      </View>

      <View style={s.actions}>
        <TouchableOpacity
          id="btn-enroll"
          style={[s.btn, s.btnPrimary]}
          onPress={onEnroll}
          activeOpacity={0.85}
        >
          <View style={s.btnIconContainerPrimary}>
            <Svg
              width={22}
              height={22}
              viewBox="0 0 24 24"
              fill="none"
              stroke="#09090B"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <Path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
              <Circle cx="9" cy="7" r="4" />
              <Line x1="19" y1="8" x2="19" y2="14" />
              <Line x1="22" y1="11" x2="16" y2="11" />
            </Svg>
          </View>
          <View>
            <Text style={s.btnLabel}>Enroll Face</Text>
            <Text style={s.btnSub}>Register a new person</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          id="btn-verify"
          style={[s.btn, s.btnOutline, names.length === 0 && s.btnDisabled]}
          onPress={names.length > 0 ? onVerify : undefined}
          activeOpacity={0.85}
        >
          <View style={s.btnIconContainerOutline}>
            <Svg
              width={22}
              height={22}
              viewBox="0 0 24 24"
              fill="none"
              stroke={C.primary}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <Path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <Path d="M9 12l2 2 4-4" />
            </Svg>
          </View>
          <View>
            <Text style={s.btnLabelOutline}>Verify Face</Text>
            <Text style={s.btnSubOutline}>
              {names.length === 0
                ? 'Enroll someone first'
                : 'Identify a person'}
            </Text>
          </View>
        </TouchableOpacity>
      </View>

      {names.length > 0 && (
        <View style={s.listSection}>
          <Text style={s.listHeader}>ENROLLED FACES</Text>
          <FlatList
            data={names}
            keyExtractor={item => item}
            renderItem={({ item }) => (
              <View style={s.faceRow}>
                <View style={s.avatar}>
                  <Text style={s.avatarText}>
                    {item.charAt(0).toUpperCase()}
                  </Text>
                </View>
                <Text style={s.faceName}>{item}</Text>
                <TouchableOpacity
                  id={`btn-delete-${item}`}
                  onPress={() => handleDelete(item)}
                  style={s.deleteBtn}
                  hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                >
                  <Svg
                    width={16}
                    height={16}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke={C.danger}
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <Path d="M3 6h18" />
                    <Path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                    <Path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </Svg>
                </TouchableOpacity>
              </View>
            )}
          />
        </View>
      )}
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
  container: { flex: 1, backgroundColor: C.surface },

  header: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 32,
    paddingHorizontal: 24,
  },
  logoRing: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: C.card,
    borderWidth: 2,
    borderColor: C.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  title: {
    color: C.text,
    fontSize: 28,
    lineHeight: 38,
    fontWeight: '800',
    letterSpacing: 0.5,
    textAlignVertical: 'center',
  },
  subtitle: {
    color: C.textSecondary,
    fontSize: 14,
    lineHeight: 22,
    marginTop: 6,
    textAlignVertical: 'center',
  },

  actions: { paddingHorizontal: 20, gap: 12 },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 18,
    borderRadius: 16,
    gap: 16,
  },
  btnPrimary: { backgroundColor: C.primary },
  btnOutline: {
    backgroundColor: C.card,
    borderWidth: 1,
    borderColor: C.border,
  },
  btnDisabled: { opacity: 0.4 },
  btnIconContainerPrimary: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#09090B15',
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnIconContainerOutline: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.cardElevated,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnLabel: {
    color: C.surface,
    fontSize: 17,
    lineHeight: 24,
    fontWeight: '700',
    textAlignVertical: 'center',
  },
  btnSub: {
    color: '#09090B99',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 2,
    textAlignVertical: 'center',
  },
  btnLabelOutline: {
    color: C.text,
    fontSize: 17,
    lineHeight: 24,
    fontWeight: '700',
    textAlignVertical: 'center',
  },
  btnSubOutline: {
    color: C.textSecondary,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 2,
    textAlignVertical: 'center',
  },

  listSection: { flex: 1, marginTop: 28, paddingHorizontal: 20 },
  listHeader: {
    color: C.textSecondary,
    fontSize: 11,
    lineHeight: 16,
    fontWeight: '700',
    letterSpacing: 1.5,
    marginBottom: 10,
  },
  faceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.card,
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    gap: 12,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: C.cardElevated,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: C.primary,
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '700',
  },
  faceName: {
    flex: 1,
    color: C.text,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '600',
  },
  deleteBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: C.danger + '15',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
