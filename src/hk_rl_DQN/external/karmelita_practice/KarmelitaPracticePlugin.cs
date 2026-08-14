using System.Collections;
using System.Collections.Generic;
using BepInEx;
using HarmonyLib;
using HutongGames.PlayMaker;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace KarmelitaPractice;

[BepInAutoPlugin(id: "io.github.hollow-knight-rl.karmelita-practice")]
public partial class KarmelitaPracticePlugin : BaseUnityPlugin
{
    private const int SaveSlot = 1;
    private const float StartupDelaySeconds = 8f;
    private const float ManagerTimeoutSeconds = 30f;
    private const float SaveLoadTimeoutSeconds = 90f;
    private const float SceneLoadTimeoutSeconds = 30f;
    private const float BossSpawnTimeoutSeconds = 15f;
    private const float DeathRestartDelaySeconds = 1f;
    private const float DeathTransitionTimeoutSeconds = 20f;
    private const float DeathSceneStableSeconds = 1f;
    private const float MenuSettleSeconds = 4f;
    private const float LoadedSceneStableSeconds = 5f;
    private const float RevealDelaySeconds = 1f;
    private const float ChallengeStartTimeoutSeconds = 3f;
    private const float ChallengeCompleteTimeoutSeconds = 12f;
    private const float PostChallengeCorrectionDelaySeconds = 0.2f;
    private const float PostCorrectionVerifySeconds = 2f;

    private bool isTransitioning;
    private bool encounterActive;
    private bool deathRestartQueued;
    private bool saveLoadCompleted;
    private bool saveLoadSucceeded;
    private bool originalDefeatedFlag;
    private bool originalFlagCaptured;
    private bool showLoadingCurtain;
    private bool pluginEnabled;
    private bool telemetryEnabled;
    private TelemetryRecorder? telemetry;
    private HeroController? subscribedHero;
    private Harmony? harmony;
    private float arenaMinX = float.NegativeInfinity;
    private float arenaMaxX = float.PositiveInfinity;
    private float safeHeroY = KarmelitaEncounter.FallbackHeroY;
    private readonly Dictionary<string, string> bossFsmStates = new();
    private int observedHeroHealth = -1;
    private Vector3 observedHeroPosition;
    private bool observedHeroPositionValid;

    internal static KarmelitaPracticePlugin? Instance { get; private set; }
    internal bool IsEncounterActive => encounterActive && !isTransitioning;
    internal Transform? ActiveBoss { get; private set; }

    private void Awake()
    {
        pluginEnabled = Config.Bind(
            "General",
            "Enabled",
            true,
            "Enable automatic Karmelita entry, F8 reload, scene patches, and death restart. Requires a game restart when changed."
        ).Value;
        if (!pluginEnabled)
        {
            Logger.LogInfo("Karmelita practice loop disabled by configuration");
            return;
        }

        Instance = this;
        telemetryEnabled = Config.Bind(
            "Telemetry",
            "Enabled",
            true,
            "Write read-only player, Boss, scene, and FSM snapshots as JSONL."
        ).Value;
        if (telemetryEnabled)
        {
            float interval = Config.Bind(
                "Telemetry",
                "IntervalSeconds",
                0.1f,
                "Seconds between telemetry snapshots."
            ).Value;
            telemetry = new TelemetryRecorder(interval);
        }
        Logger.LogInfo("Karmelita practice loop ready");
        harmony = new Harmony("io.github.hollow-knight-rl.karmelita-practice");
        harmony.PatchAll(typeof(KarmelitaScenePatches).Assembly);
        StartCoroutine(AutoEnterArena());
    }

    private void Update()
    {
        if (!pluginEnabled || isTransitioning)
        {
            return;
        }

        telemetry?.Tick(this);

        if (Input.GetKeyDown(KeyCode.F8))
        {
            StartCoroutine(EnterArena(true));
        }

    }

    private void OnGUI()
    {
        if (!showLoadingCurtain)
        {
            return;
        }

        GUI.depth = int.MinValue;
        GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture, ScaleMode.StretchToFill, false, 0f, Color.black, 0f, 0f);
    }

    private IEnumerator AutoEnterArena()
    {
        yield return new WaitForSecondsRealtime(StartupDelaySeconds);

        float deadline = Time.realtimeSinceStartup + ManagerTimeoutSeconds;
        while (GameManager.instance == null && Time.realtimeSinceStartup < deadline)
        {
            yield return null;
        }

        if (GameManager.instance == null)
        {
            Logger.LogError("Automatic entry stopped: GameManager did not become ready");
            yield break;
        }

        isTransitioning = true;
        showLoadingCurtain = true;
        Logger.LogInfo($"Automatically loading save slot {SaveSlot}");
        GameManager.instance.LoadGame(
            SaveSlot,
            succeeded =>
            {
                saveLoadSucceeded = succeeded;
                saveLoadCompleted = true;
            }
        );

        deadline = Time.realtimeSinceStartup + SaveLoadTimeoutSeconds;
        while (!saveLoadCompleted && Time.realtimeSinceStartup < deadline)
        {
            yield return null;
        }

        if (!saveLoadCompleted || !saveLoadSucceeded)
        {
            isTransitioning = false;
            Logger.LogError($"Automatic entry stopped: save slot {SaveSlot} could not be read");
            yield break;
        }

        // The title screen still has a profile-menu coroutine in flight when
        // LoadGame's callback fires. Let it finish before ContinueGame tears it down.
        yield return new WaitForSecondsRealtime(MenuSettleSeconds);
        Logger.LogInfo($"Save slot {SaveSlot} loaded; continuing game");
        GameManager.instance.ContinueGame();

        deadline = Time.realtimeSinceStartup + SaveLoadTimeoutSeconds;
        while (
            (HeroController.instance == null || GameManager.instance.IsInSceneTransition)
            && Time.realtimeSinceStartup < deadline
        )
        {
            yield return null;
        }

        if (HeroController.instance == null || GameManager.instance.IsInSceneTransition)
        {
            isTransitioning = false;
            Logger.LogError("Automatic entry stopped: loaded game did not become playable");
            yield break;
        }

        string loadedScene = SceneManager.GetActiveScene().name;
        float stableSince = Time.realtimeSinceStartup;
        deadline = Time.realtimeSinceStartup + SaveLoadTimeoutSeconds;
        while (Time.realtimeSinceStartup - stableSince < LoadedSceneStableSeconds && Time.realtimeSinceStartup < deadline)
        {
            if (
                HeroController.instance == null
                || GameManager.instance.IsInSceneTransition
                || !GameManager.instance.HasFinishedEnteringScene
                || SceneManager.GetActiveScene().name != loadedScene
            )
            {
                loadedScene = SceneManager.GetActiveScene().name;
                stableSince = Time.realtimeSinceStartup;
            }
            yield return null;
        }
        isTransitioning = false;
        yield return EnterArena(true);
        if (encounterActive)
        {
            Logger.LogInfo("Automatic Karmelita entry complete");
        }
    }

    private IEnumerator RestartAfterDeath()
    {
        Logger.LogInfo("Hornet died; restarting Karmelita encounter");
        showLoadingCurtain = true;
        yield return new WaitForSecondsRealtime(DeathRestartDelaySeconds);

        float deadline = Time.realtimeSinceStartup + DeathTransitionTimeoutSeconds;
        string deathScene = SceneManager.GetActiveScene().name;
        while (
            GameManager.instance != null
            && !GameManager.instance.IsInSceneTransition
            && SceneManager.GetActiveScene().name == deathScene
            && Time.realtimeSinceStartup < deadline
        )
        {
            yield return null;
        }

        while (
            GameManager.instance != null
            && GameManager.instance.IsInSceneTransition
            && Time.realtimeSinceStartup < deadline
        )
        {
            yield return null;
        }

        string stableScene = SceneManager.GetActiveScene().name;
        float stableSince = Time.realtimeSinceStartup;
        while (
            Time.realtimeSinceStartup - stableSince < DeathSceneStableSeconds
            && Time.realtimeSinceStartup < deadline
        )
        {
            if (
                GameManager.instance == null
                || GameManager.instance.IsInSceneTransition
                || SceneManager.GetActiveScene().name != stableScene
            )
            {
                stableScene = SceneManager.GetActiveScene().name;
                stableSince = Time.realtimeSinceStartup;
            }
            yield return null;
        }

        Logger.LogInfo($"Death flow settled in {SceneManager.GetActiveScene().name}; reloading save slot {SaveSlot}");
        GameManager? gameManager = GameManager.instance;
        if (gameManager == null)
        {
            Logger.LogError("Death restart stopped: GameManager disappeared");
            deathRestartQueued = false;
            showLoadingCurtain = false;
            yield break;
        }
        saveLoadCompleted = false;
        saveLoadSucceeded = false;
        gameManager.LoadGame(
            SaveSlot,
            succeeded =>
            {
                saveLoadSucceeded = succeeded;
                saveLoadCompleted = true;
            }
        );

        deadline = Time.realtimeSinceStartup + SaveLoadTimeoutSeconds;
        while (!saveLoadCompleted && Time.realtimeSinceStartup < deadline)
        {
            yield return null;
        }

        if (!saveLoadCompleted || !saveLoadSucceeded)
        {
            Logger.LogError($"Death restart stopped: save slot {SaveSlot} could not be read");
            deathRestartQueued = false;
            showLoadingCurtain = false;
            yield break;
        }

        gameManager.ContinueGame();
        deadline = Time.realtimeSinceStartup + SaveLoadTimeoutSeconds;
        while (
            (HeroController.instance == null || gameManager.IsInSceneTransition || !gameManager.HasFinishedEnteringScene)
            && Time.realtimeSinceStartup < deadline
        )
        {
            yield return null;
        }

        stableScene = SceneManager.GetActiveScene().name;
        stableSince = Time.realtimeSinceStartup;
        while (
            Time.realtimeSinceStartup - stableSince < LoadedSceneStableSeconds
            && Time.realtimeSinceStartup < deadline
        )
        {
            if (
                HeroController.instance == null
                || gameManager.IsInSceneTransition
                || !gameManager.HasFinishedEnteringScene
                || SceneManager.GetActiveScene().name != stableScene
            )
            {
                stableScene = SceneManager.GetActiveScene().name;
                stableSince = Time.realtimeSinceStartup;
            }
            yield return null;
        }

        Logger.LogInfo($"Save slot {SaveSlot} reloaded cleanly in {stableScene}; entering challenge");
        yield return EnterArena(true);
        deathRestartQueued = false;
    }

    private IEnumerator EnterArena(bool forceReload)
    {
        if (isTransitioning || GameManager.instance == null)
        {
            yield break;
        }

        if (!forceReload && encounterActive && SceneManager.GetActiveScene().name == KarmelitaEncounter.SceneName)
        {
            yield break;
        }

        isTransitioning = true;
        encounterActive = false;
        ActiveBoss = null;
        arenaMinX = float.NegativeInfinity;
        arenaMaxX = float.PositiveInfinity;
        showLoadingCurtain = true;
        PlayerData? playerData = PlayerData.instance;
        if (playerData != null)
        {
            if (!originalFlagCaptured)
            {
                originalDefeatedFlag = playerData.GetBool(KarmelitaEncounter.DefeatedFlag);
                originalFlagCaptured = true;
            }
            playerData.SetBool(KarmelitaEncounter.DefeatedFlag, false);
            playerData.SetInt("health", playerData.GetInt("maxHealth"));
        }

        Logger.LogInfo($"Entering {KarmelitaEncounter.DisplayName}");
        GameManager.instance.BeginSceneTransition(new GameManager.SceneLoadInfo
        {
            SceneName = KarmelitaEncounter.SceneName,
            EntryGateName = KarmelitaEncounter.EntryGateName,
            PreventCameraFadeOut = false,
            WaitForSceneTransitionCameraFade = true,
            Visualization = GameManager.SceneLoadVisualizations.Default
        });

        float deadline = Time.realtimeSinceStartup + SceneLoadTimeoutSeconds;
        while (SceneManager.GetActiveScene().name != KarmelitaEncounter.SceneName && Time.realtimeSinceStartup < deadline)
        {
            yield return null;
        }

        if (SceneManager.GetActiveScene().name != KarmelitaEncounter.SceneName)
        {
            isTransitioning = false;
            Logger.LogError("Karmelita scene transition timed out; press F8 to retry");
            yield break;
        }

        deadline = Time.realtimeSinceStartup + SceneLoadTimeoutSeconds;
        while (
            (HeroController.instance == null || GameManager.instance.IsInSceneTransition || !GameManager.instance.HasFinishedEnteringScene)
            && Time.realtimeSinceStartup < deadline
        )
        {
            yield return null;
        }

        deadline = Time.realtimeSinceStartup + BossSpawnTimeoutSeconds;
        GameObject? boss;
        do
        {
            boss = GameObject.Find(KarmelitaEncounter.BossPath);
            if (boss == null)
            {
                yield return null;
            }
        }
        while (boss == null && Time.realtimeSinceStartup < deadline);

        encounterActive = boss != null;
        if (encounterActive)
        {
            ActiveBoss = boss!.transform;
            Logger.LogInfo($"Karmelita boss found: {boss!.name}");
            StartCoroutine(MonitorBossIntent());
            PlayMakerFSM? challengeFsm = FindChallengeFsm();
            if (challengeFsm == null)
            {
                Logger.LogError("Karmelita challenge FSM was not found");
                isTransitioning = false;
                yield break;
            }

            Logger.LogInfo($"Starting challenge through {GetPath(challengeFsm.transform)} / {challengeFsm.FsmName} ({challengeFsm.ActiveStateName})");
            challengeFsm.SendEvent("SPECIAL CHALLENGE");

            deadline = Time.realtimeSinceStartup + ChallengeStartTimeoutSeconds;
            while (
                challengeFsm != null
                && challengeFsm.ActiveStateName == "Idle"
                && Time.realtimeSinceStartup < deadline
            )
            {
                yield return null;
            }

            deadline = Time.realtimeSinceStartup + ChallengeCompleteTimeoutSeconds;
            while (
                challengeFsm != null
                && challengeFsm.ActiveStateName != "Challenge Complete"
                && Time.realtimeSinceStartup < deadline
            )
            {
                yield return null;
            }
            Logger.LogInfo($"Karmelita challenge state: {challengeFsm?.ActiveStateName ?? "destroyed"}");
            if (challengeFsm != null && challengeFsm.ActiveStateName == "Idle")
            {
                Logger.LogError("SPECIAL CHALLENGE was ignored; revealing scene for manual recovery");
            }
            SubscribeToHeroDeath();
            yield return CorrectAndVerifyCombatPosition(boss.transform);
            yield return new WaitForSecondsRealtime(RevealDelaySeconds);
            isTransitioning = false;
            showLoadingCurtain = false;
        }
        else
        {
            isTransitioning = false;
            Logger.LogError("Karmelita scene loaded, but Hunter Queen Boss did not spawn");
        }
    }

    private IEnumerator MonitorBossIntent()
    {
        observedHeroHealth = PlayerData.instance != null ? PlayerData.instance.GetInt("health") : -1;
        while (encounterActive && ActiveBoss != null && SceneManager.GetActiveScene().name == KarmelitaEncounter.SceneName)
        {
            if (PlayerData.instance != null)
            {
                int health = PlayerData.instance.GetInt("health");
                if (observedHeroHealth >= 0 && health < observedHeroHealth)
                {
                    Logger.LogInfo($"Karmelita hero health changed: {observedHeroHealth} -> {health}; first-hit trigger ready");
                }
                observedHeroHealth = health;
            }
            HeroController? hero = HeroController.instance;
            if (hero != null)
            {
                Vector3 position = hero.transform.position;
                Rigidbody2D? body = hero.GetComponent<Rigidbody2D>();
                Vector2 velocity = body != null ? body.linearVelocity : Vector2.zero;
                if (!observedHeroPositionValid || Vector3.Distance(position, observedHeroPosition) >= 0.05f)
                {
                    Logger.LogInfo($"Karmelita hero position: ({position.x:F2}, {position.y:F2}, {position.z:F2}), velocity=({velocity.x:F2}, {velocity.y:F2})");
                    observedHeroPosition = position;
                    observedHeroPositionValid = true;
                }
            }
            foreach (PlayMakerFSM fsm in Resources.FindObjectsOfTypeAll<PlayMakerFSM>())
            {
                if (!fsm.gameObject.scene.IsValid() || fsm.gameObject.scene.name != KarmelitaEncounter.SceneName)
                {
                    continue;
                }
                string path = GetPath(fsm.transform);
                string state = fsm.ActiveStateName ?? "<none>";
                if (!bossFsmStates.TryGetValue(path, out string? previous) || previous != state)
                {
                    bossFsmStates[path] = state;
                    Logger.LogInfo($"Karmelita boss intent: {path} / {fsm.FsmName} = {state}");
                }
            }
            yield return new WaitForSecondsRealtime(0.1f);
        }
        bossFsmStates.Clear();
    }

    private void PlaceHeroOnCombatSide(Vector3 bossPosition, bool logPosition)
    {
        HeroController? hero = HeroController.instance;
        if (hero == null)
        {
            return;
        }

        float x = bossPosition.x + KarmelitaEncounter.CombatSideOffsetX;
        float y = FindSafeHeroY(hero, x);
        safeHeroY = y;
        MeasureArenaBounds(hero, bossPosition);
        hero.transform.position = new Vector3(x, y, hero.transform.position.z);
        hero.FaceLeft();
        if (logPosition)
        {
            Logger.LogInfo($"Karmelita combat position: Hornet={hero.transform.position}, Boss={bossPosition}");
        }
    }

    private void MeasureArenaBounds(HeroController hero, Vector3 bossPosition)
    {
        Collider2D? heroCollider = hero.GetComponent<Collider2D>();
        float heroHalfWidth = heroCollider != null ? heroCollider.bounds.extents.x : 0.5f;
        float probeY = KarmelitaEncounter.ArenaGroundY + 2f;
        arenaMinX = FindSolidWallX(new Vector2(bossPosition.x, probeY), Vector2.left, 40f, bossPosition.x - 18f)
            + heroHalfWidth + KarmelitaEncounter.WallSafetyMargin;
        arenaMaxX = FindSolidWallX(new Vector2(bossPosition.x, probeY), Vector2.right, 40f, bossPosition.x + 18f)
            - heroHalfWidth - KarmelitaEncounter.WallSafetyMargin;
        Logger.LogInfo($"Karmelita arena bounds: {arenaMinX:F2} to {arenaMaxX:F2}");
    }

    private float FindSolidWallX(Vector2 origin, Vector2 direction, float distance, float fallback)
    {
        foreach (RaycastHit2D hit in Physics2D.RaycastAll(origin, direction, distance))
        {
            if (
                hit.collider == null
                || hit.collider.isTrigger
                || (ActiveBoss != null && hit.collider.transform.IsChildOf(ActiveBoss))
                || (HeroController.instance != null && hit.collider.transform.IsChildOf(HeroController.instance.transform))
            )
            {
                continue;
            }
            return hit.point.x;
        }
        return fallback;
    }

    private void KeepHeroInsideArena()
    {
        if (!IsEncounterActive || HeroController.instance == null)
        {
            return;
        }

        Transform hero = HeroController.instance.transform;
        Vector3 position = hero.position;
        float clampedX = position.x;
        if (position.x < arenaMinX - KarmelitaEncounter.BoundaryTolerance)
        {
            clampedX = arenaMinX + KarmelitaEncounter.WallRecoveryInset;
        }
        else if (position.x > arenaMaxX + KarmelitaEncounter.BoundaryTolerance)
        {
            clampedX = arenaMaxX - KarmelitaEncounter.WallRecoveryInset;
        }
        float clampedY = position.y < KarmelitaEncounter.ArenaGroundY - 1f ? safeHeroY : position.y;
        if (Mathf.Approximately(clampedX, position.x) && Mathf.Approximately(clampedY, position.y))
        {
            return;
        }

        hero.position = new Vector3(clampedX, clampedY, position.z);
        Rigidbody2D? body = HeroController.instance.GetComponent<Rigidbody2D>();
        if (body != null)
        {
            Vector2 velocity = body.linearVelocity;
            if (clampedX != position.x)
            {
                velocity.x = 0f;
            }
            if (clampedY != position.y)
            {
                velocity.y = 0f;
            }
            body.linearVelocity = velocity;
        }
        Logger.LogWarning($"Recovered Hornet from outside arena: {position} -> {hero.position}");
    }

    private float FindSafeHeroY(HeroController hero, float x)
    {
        Collider2D? heroCollider = hero.GetComponent<Collider2D>();
        float halfHeight = heroCollider != null ? heroCollider.bounds.extents.y : 0f;
        float localCenterOffset = heroCollider != null ? hero.transform.position.y - heroCollider.bounds.center.y : 0f;
        RaycastHit2D[] hits = Physics2D.RaycastAll(new Vector2(x, 30f), Vector2.down, 30f);
        foreach (RaycastHit2D hit in hits)
        {
            if (
                hit.collider == null
                || hit.collider.isTrigger
                || hit.collider.transform.IsChildOf(hero.transform)
                || (ActiveBoss != null && hit.collider.transform.IsChildOf(ActiveBoss))
            )
            {
                continue;
            }

            float y = hit.point.y + halfHeight + localCenterOffset + 0.02f;
            Logger.LogInfo($"Karmelita ground: {hit.collider.name} at {hit.point}, heroY={y:F2}");
            return y;
        }

        Logger.LogWarning("No solid Karmelita ground found; using measured fallback height");
        return KarmelitaEncounter.FallbackHeroY;
    }

    private IEnumerator CorrectAndVerifyCombatPosition(Transform boss)
    {
        yield return new WaitForSecondsRealtime(PostChallengeCorrectionDelaySeconds);
        if (boss != null && HeroController.instance != null)
        {
            Rigidbody2D? body = HeroController.instance.GetComponent<Rigidbody2D>();
            if (body != null)
            {
                body.linearVelocity = Vector2.zero;
            }
            PlaceHeroOnCombatSide(boss.position, true);
            Logger.LogInfo("Applied post-challenge combat position correction");
        }

        yield return new WaitForSecondsRealtime(PostCorrectionVerifySeconds);
        if (HeroController.instance != null)
        {
            Logger.LogInfo($"Karmelita settled position: Hornet={HeroController.instance.transform.position}, Scene={SceneManager.GetActiveScene().name}");
        }
    }

    private void SubscribeToHeroDeath()
    {
        if (HeroController.instance == null || subscribedHero == HeroController.instance)
        {
            return;
        }

        if (subscribedHero != null)
        {
            subscribedHero.OnDeath -= QueueDeathRestart;
        }
        subscribedHero = HeroController.instance;
        subscribedHero.OnDeath += QueueDeathRestart;
    }

    private void QueueDeathRestart()
    {
        if (!encounterActive || deathRestartQueued)
        {
            return;
        }

        deathRestartQueued = true;
        encounterActive = false;
        ActiveBoss = null;
        showLoadingCurtain = true;
        StartCoroutine(RestartAfterDeath());
    }

    private PlayMakerFSM? FindChallengeFsm()
    {
        foreach (PlayMakerFSM fsm in Resources.FindObjectsOfTypeAll<PlayMakerFSM>())
        {
            if (
                fsm.gameObject.scene.IsValid()
                && fsm.gameObject.scene.name == KarmelitaEncounter.SceneName
                && fsm.FsmStates != null
                && System.Array.Exists(fsm.FsmStates, state => state.Name == "Challenge 1")
                && System.Array.Exists(fsm.FsmStates, state => state.Name == "Challenge 2")
            )
            {
                return fsm;
            }
        }
        return null;
    }

    private static string GetPath(Transform transform)
    {
        string path = transform.name;
        while (transform.parent != null)
        {
            transform = transform.parent;
            path = $"{transform.name}/{path}";
        }
        return path;
    }

    private void OnDestroy()
    {
        telemetry?.Dispose();
        telemetry = null;
        Instance = null;
        if (subscribedHero != null)
        {
            subscribedHero.OnDeath -= QueueDeathRestart;
        }
        harmony?.UnpatchSelf();
        if (originalFlagCaptured && PlayerData.instance != null)
        {
            PlayerData.instance.SetBool(KarmelitaEncounter.DefeatedFlag, originalDefeatedFlag);
        }
    }
}
