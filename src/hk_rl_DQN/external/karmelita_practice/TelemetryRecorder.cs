using System;
using System.Globalization;
using System.IO;
using System.Text;
using BepInEx;
using HutongGames.PlayMaker;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace KarmelitaPractice;

internal sealed class TelemetryRecorder : IDisposable
{
    private readonly StreamWriter writer;
    private readonly float intervalSeconds;
    private float nextSampleTime;
    private long sequence;

    internal TelemetryRecorder(float intervalSeconds)
    {
        this.intervalSeconds = Mathf.Max(0.01f, intervalSeconds);
        string directory = Path.Combine(Paths.PluginPath, "hollow-knight-rl-KarmelitaPractice");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "telemetry.jsonl");
        writer = new StreamWriter(path, append: true, Encoding.UTF8) { AutoFlush = true };
        writer.WriteLine($"{{\"type\":\"telemetry_start\",\"timestamp\":{Number(Time.realtimeSinceStartup)}}}");
    }

    internal void Tick(KarmelitaPracticePlugin plugin)
    {
        if (Time.realtimeSinceStartup < nextSampleTime)
        {
            return;
        }
        nextSampleTime = Time.realtimeSinceStartup + intervalSeconds;
        WriteSnapshot(plugin);
    }

    private void WriteSnapshot(KarmelitaPracticePlugin plugin)
    {
        HeroController? hero = HeroController.instance;
        Transform? boss = plugin.ActiveBoss;
        Rigidbody2D? heroBody = hero != null ? hero.GetComponent<Rigidbody2D>() : null;
        Rigidbody2D? bossBody = boss != null ? boss.GetComponent<Rigidbody2D>() : null;
        string scene = SceneManager.GetActiveScene().name;
        StringBuilder json = new();
        json.Append("{\"type\":\"snapshot\"");
        json.Append($",\"sequence\":{sequence++},\"timestamp\":{Number(Time.realtimeSinceStartup)}");
        json.Append($",\"frame\":{Time.frameCount},\"scene\":{Quote(scene)}");
        json.Append($",\"encounter_active\":{Bool(plugin.IsEncounterActive)}");
        json.Append(",\"player_grounded\":");
        json.Append(hero != null ? Bool(hero.cState.onGround) : "null");
        json.Append(",\"player\":");
        AppendTransform(json, hero != null ? hero.transform : null, heroBody);
        json.Append(",\"player_health\":");
        if (PlayerData.instance == null)
        {
            json.Append("null");
        }
        else
        {
            json.Append($"{{\"health\":{PlayerData.instance.GetInt("health")},\"max_health\":{PlayerData.instance.GetInt("maxHealth")}}}");
        }
        json.Append(",\"player_resources\":");
        AppendPlayerResources(json, hero, PlayerData.instance);
        json.Append(",\"player_control\":");
        AppendPlayerControl(json, hero);
        json.Append(",\"boss\":");
        AppendTransform(json, boss, bossBody);
        json.Append(",\"fsm\":[");
        bool first = true;
        foreach (PlayMakerFSM fsm in Resources.FindObjectsOfTypeAll<PlayMakerFSM>())
        {
            if (!fsm.gameObject.scene.IsValid() || fsm.gameObject.scene.name != scene)
            {
                continue;
            }
            bool belongsToBoss = boss != null
                && (fsm.transform == boss || fsm.transform.IsChildOf(boss));
            bool isChallengeState = fsm.gameObject.name == "Challenge Region"
                && fsm.FsmName == "Challenge";
            if (!belongsToBoss && !isChallengeState)
            {
                continue;
            }
            if (!first) json.Append(',');
            first = false;
            json.Append($"{{\"path\":{Quote(GetPath(fsm.transform))},\"name\":{Quote(fsm.FsmName)},\"state\":{Quote(fsm.ActiveStateName ?? "<none>")}}}");
        }
        json.Append("]}");
        writer.WriteLine(json.ToString());
    }

    private static void AppendPlayerResources(
        StringBuilder json,
        HeroController? hero,
        PlayerData? playerData)
    {
        if (playerData == null)
        {
            json.Append("null");
            return;
        }
        int skillCost = playerData.SilkSkillCost;
        bool silkAbilitiesDisabled = playerData.disableSilkAbilities;
        bool skillAvailable = hero != null && hero.CanHarpoonDash();
        bool spellAvailable = hero != null
            && !silkAbilitiesDisabled
            && playerData.silk >= skillCost
            && hero.CanThrowTool(false);
        json.Append($"{{\"silk\":{playerData.silk}");
        json.Append($",\"silk_max\":{playerData.CurrentSilkMax}");
        json.Append($",\"silk_parts\":{playerData.silkParts}");
        json.Append($",\"skill_cost\":{skillCost}");
        json.Append($",\"silk_abilities_disabled\":{Bool(silkAbilitiesDisabled)}");
        json.Append($",\"skill_available\":{Bool(skillAvailable)}");
        json.Append($",\"spell_available\":{Bool(spellAvailable)}}}");
    }

    private static void AppendPlayerControl(StringBuilder json, HeroController? hero)
    {
        if (hero == null)
        {
            json.Append("null");
            return;
        }
        json.Append($"{{\"jump_available\":{Bool(hero.CanJump())}");
        json.Append($",\"dash_available\":{Bool(hero.CanDash())}");
        json.Append($",\"attack_available\":{Bool(hero.CanAttack())}}}");
    }

    private static void AppendTransform(StringBuilder json, Transform? transform, Rigidbody2D? body)
    {
        if (transform == null)
        {
            json.Append("null");
            return;
        }
        Vector3 position = transform.position;
        Vector2 velocity = body != null ? body.linearVelocity : Vector2.zero;
        json.Append($"{{\"x\":{Number(position.x)},\"y\":{Number(position.y)},\"z\":{Number(position.z)},\"velocity_x\":{Number(velocity.x)},\"velocity_y\":{Number(velocity.y)}}}");
    }

    private static string Number(float value) => value.ToString("R", CultureInfo.InvariantCulture);
    private static string Bool(bool value) => value ? "true" : "false";

    private static string Quote(string value)
    {
        StringBuilder escaped = new(value.Length + 2);
        escaped.Append('"');
        foreach (char character in value)
        {
            switch (character)
            {
                case '\\': escaped.Append("\\\\"); break;
                case '"': escaped.Append("\\\""); break;
                case '\n': escaped.Append("\\n"); break;
                case '\r': escaped.Append("\\r"); break;
                case '\t': escaped.Append("\\t"); break;
                default: escaped.Append(character); break;
            }
        }
        escaped.Append('"');
        return escaped.ToString();
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

    public void Dispose()
    {
        writer.WriteLine($"{{\"type\":\"telemetry_stop\",\"timestamp\":{Number(Time.realtimeSinceStartup)}}}");
        writer.Dispose();
    }
}
