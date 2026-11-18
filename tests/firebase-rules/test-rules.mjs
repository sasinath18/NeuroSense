import { readFile } from "node:fs/promises";
import {
  assertFails,
  initializeTestEnvironment,
} from "@firebase/rules-unit-testing";
import { deleteObject, getBytes, ref, uploadBytes } from "firebase/storage";
import { doc, getDoc, setDoc } from "firebase/firestore";

const projectId = "demo-neurosense-rules";
const firestoreRules = await readFile(
  new URL("../../firestore.rules", import.meta.url),
  "utf8",
);
const storageRules = await readFile(
  new URL("../../storage.rules", import.meta.url),
  "utf8",
);

const environment = await initializeTestEnvironment({
  projectId,
  firestore: { rules: firestoreRules },
  storage: { rules: storageRules },
});

let passed = 0;

async function expectDenied(label, operation) {
  await assertFails(operation);
  passed += 1;
  console.log(`PASS: ${label}`);
}

try {
  const anonymous = environment.unauthenticatedContext();
  const authenticated = environment.authenticatedContext("rules-test-user");
  const firestorePath = ["users", "rules-test-user", "private", "probe"];
  const objectPath = "users/rules-test-user/music/protected.mp3";
  const seedBytes = new Uint8Array([78, 101, 117, 114, 111, 83, 101, 110, 115, 101]);

  // Seed one object through the test harness so read/delete checks exercise
  // existing data rather than depending on a missing-object response.
  await environment.withSecurityRulesDisabled(async (adminContext) => {
    await uploadBytes(ref(adminContext.storage(), objectPath), seedBytes, {
      contentType: "audio/mpeg",
    });
  });

  for (const [label, context] of [
    ["unauthenticated client", anonymous],
    ["authenticated client", authenticated],
  ]) {
    const firestore = context.firestore();
    const storage = context.storage();

    await expectDenied(
      `${label}: Firestore document read`,
      getDoc(doc(firestore, ...firestorePath)),
    );
    await expectDenied(
      `${label}: Firestore document write`,
      setDoc(doc(firestore, ...firestorePath), { shouldBeDenied: true }),
    );
    await expectDenied(
      `${label}: Storage object read`,
      getBytes(ref(storage, objectPath)),
    );
    await expectDenied(
      `${label}: Storage object upload`,
      uploadBytes(ref(storage, objectPath), seedBytes, {
        contentType: "audio/mpeg",
      }),
    );
    await expectDenied(
      `${label}: Storage object delete`,
      deleteObject(ref(storage, objectPath)),
    );
  }

  console.log(`Firebase Security Rules Emulator: ${passed} client access checks denied as expected.`);
} finally {
  await environment.cleanup();
}
