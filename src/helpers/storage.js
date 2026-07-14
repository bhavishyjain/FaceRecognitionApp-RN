import AsyncStorage from '@react-native-async-storage/async-storage';

const PREFIX = 'face_embedding_';
const NAMES_KEY = 'face_names';

export async function saveEmbedding(name, embedding) {
  await AsyncStorage.setItem(
    `${PREFIX}${name}`,
    JSON.stringify(Array.from(embedding)),
  );
  const names = await loadEnrolledNames();
  if (!names.includes(name)) {
    await AsyncStorage.setItem(NAMES_KEY, JSON.stringify([...names, name]));
  }
}

export async function loadEnrolledNames() {
  try {
    const raw = await AsyncStorage.getItem(NAMES_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function loadEmbedding(name) {
  try {
    const raw = await AsyncStorage.getItem(`${PREFIX}${name}`);
    return raw ? new Float32Array(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export async function loadAllEmbeddings() {
  const names = await loadEnrolledNames();
  const results = [];
  for (const name of names) {
    const embedding = await loadEmbedding(name);
    if (embedding) {
      results.push({ name, embedding });
    }
  }
  return results;
}

export async function deleteEmbedding(name) {
  await AsyncStorage.removeItem(`${PREFIX}${name}`);
  const names = await loadEnrolledNames();
  await AsyncStorage.setItem(
    NAMES_KEY,
    JSON.stringify(names.filter(n => n !== name)),
  );
}
