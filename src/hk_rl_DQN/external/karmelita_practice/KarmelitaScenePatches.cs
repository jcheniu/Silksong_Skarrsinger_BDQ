using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace KarmelitaPractice;

[HarmonyPatch]
internal static class KarmelitaScenePatches
{
    [HarmonyPrefix]
    [HarmonyPatch(typeof(CameraTarget), nameof(CameraTarget.Update))]
    private static bool FollowHeroAndKarmelita(CameraTarget __instance)
    {
        KarmelitaPracticePlugin? plugin = KarmelitaPracticePlugin.Instance;
        Transform? boss = plugin?.ActiveBoss;
        if (
            plugin == null
            || !plugin.IsEncounterActive
            || boss == null
            || HeroController.instance == null
            || SceneManager.GetActiveScene().name != KarmelitaEncounter.SceneName
        )
        {
            return true;
        }

        Vector3 hero = HeroController.instance.transform.position;
        Vector3 position = __instance.transform.position;
        position.x = Mathf.Clamp((hero.x + boss.position.x) * 0.5f, __instance.xLockMin, __instance.xLockMax);
        position.y = Mathf.Clamp((hero.y + boss.position.y) * 0.5f, __instance.yLockMin, __instance.yLockMax);
        __instance.transform.position = position;
        return false;
    }

    [HarmonyPostfix]
    [HarmonyPatch(typeof(BattleScene), nameof(BattleScene.StartBattle))]
    private static void SkipPreludeBattle(BattleScene __instance)
    {
        if (SceneManager.GetActiveScene().name != KarmelitaEncounter.SceneName)
        {
            return;
        }

        __instance.endScene.SendEvent("BATTLE END");
        __instance.gameObject.SetActive(false);
    }
}
