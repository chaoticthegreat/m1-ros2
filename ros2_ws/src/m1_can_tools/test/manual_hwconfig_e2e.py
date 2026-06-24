"""Drive the m1_hwconfig page with Playwright and screenshot every function."""
import os, time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8095/"
OUT = "/tmp/m1_shots"
os.makedirs(OUT, exist_ok=True)
shots = []

def shot(page, name):
    p = os.path.join(OUT, name + ".png")
    page.screenshot(path=p, full_page=True)
    shots.append(p)
    print("SHOT", p, flush=True)

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1320, "height": 1100}, device_scale_factor=2)
    page.goto(URL, wait_until="domcontentloaded")

    # Wait for telemetry to populate (17 rows) and the bus to read ready.
    page.wait_for_function("document.querySelectorAll('#telBody tr').length >= 17", timeout=15000)
    page.wait_for_function("document.getElementById('statusText').textContent.includes('bus ready')", timeout=15000)
    # let a few telemetry polls land so pos/temp show real values
    page.wait_for_function(
        "[...document.querySelectorAll('#telBody tr')].filter(r=>r.cells[6].textContent!=='–').length>=17",
        timeout=15000)
    time.sleep(0.6)
    shot(page, "01_overview")

    # 2) SCAN the bus (1..17) -> responders listed
    page.fill("#scanFrom", "1"); page.fill("#scanTo", "17")
    page.click("#scanBtn")
    page.wait_for_function("document.getElementById('scanResult').textContent.includes('Found')", timeout=8000)
    time.sleep(0.3); shot(page, "02_scan")

    # 3) LIMITS editor -> edit + save -> joint_limits.yaml
    page.select_option("#limJoint", "openarm_left_joint3")
    time.sleep(0.2)
    page.fill("#limPosLo", "-1.8"); page.fill("#limPosHi", "1.8")
    page.fill("#limVel", "5"); page.fill("#limEff", "12")
    page.click("#limSave")
    page.wait_for_function("document.getElementById('limMsg').textContent.includes('saved')", timeout=8000)
    time.sleep(0.3); shot(page, "03_limits")

    # 4) JOG / TEST a motor (enable, then hold-to-jog to pos 1.2)
    page.select_option("#jogJoint", "openarm_left_joint1")
    page.click("#enBtn")
    page.eval_on_selector("#jogPos", "el=>{el.value='1.20'; el.dispatchEvent(new Event('input',{bubbles:true}))}")
    time.sleep(0.2)
    page.dispatch_event("#jogHold", "mousedown")
    page.wait_for_function(
        "[...document.querySelectorAll('#telBody tr')].some(r=>r.cells[0].textContent==='openarm_left_joint1' "
        "&& Math.abs(parseFloat(r.cells[3].textContent)-1.2)<0.06 && r.cells[9].textContent.includes('jogging'))",
        timeout=8000)
    time.sleep(0.3); shot(page, "04_jog")
    page.dispatch_event("#jogHold", "mouseup")
    # Disable to clear the server dead-man so it won't re-jog over the zero, then
    # let the dead-man window lapse.
    page.click("#disBtn")
    time.sleep(0.9)

    # 5) SET ZERO -> commanded position returns to 0
    page.click("#zeroBtn")
    page.wait_for_function("document.getElementById('jogMsg').textContent.includes('zeroed')", timeout=8000)
    page.wait_for_function(
        "[...document.querySelectorAll('#telBody tr')].some(r=>r.cells[0].textContent==='openarm_left_joint1' "
        "&& Math.abs(parseFloat(r.cells[3].textContent))<0.06)",
        timeout=8000)
    time.sleep(0.3); shot(page, "05_setzero")

    # 6) ASSIGN & MAP -> map a joint (model change) + reassign a CAN id
    page.select_option("#mapJoint", "openarm_right_joint5")
    time.sleep(0.2)
    page.select_option("#mapModel", "DM8009")
    page.click("#mapBtn")
    page.wait_for_function("document.getElementById('mapMsg').textContent.includes('mapped')", timeout=8000)
    page.fill("#asgOld", "17"); page.fill("#asgNew", "32")
    page.click("#asgBtn")
    page.wait_for_function("document.getElementById('asgMsg').textContent.includes('reassigned')", timeout=8000)
    time.sleep(0.4); shot(page, "06_assign_map")

    # 7) RUN MODE (read-only): banner flips, write controls disabled
    page.click("#modeRun")
    page.wait_for_function("document.getElementById('ownerText').textContent.includes('run')", timeout=8000)
    page.wait_for_function("document.getElementById('scanBtn').disabled === true", timeout=8000)
    time.sleep(0.4); shot(page, "07_runmode")
    page.click("#modeMaint")

    b.close()

print("ALL_SHOTS", len(shots), flush=True)
for s in shots:
    print(s, os.path.getsize(s), flush=True)
