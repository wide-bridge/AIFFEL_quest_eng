from __future__ import annotations


def get_ui_html() -> str:
    """
    Return a self-contained HTML page for BabyCoach PoC.

    Notes:
    - UI is status-check UX (chips/sliders/toggles).
    - Images are optional and use `/assets/...` with `onerror` fallback.
    """

    # PoC 2차: template-based UI (light theme + tabs).
    # This avoids editing huge inline HTML strings.
    try:
        from pathlib import Path

        template_path = Path(__file__).with_name("babycoach_ui.html")
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
    except Exception:
        # Fallback to the legacy inline UI below.
        pass

    # Keep HTML/JS as a single string to simplify PoC setup.
    return r"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BabyCoach PoC</title>
    <style>
      :root{
        --bg:#0b1220;
        --card:#121c33;
        --card2:#0f1830;
        --text:#e8eefc;
        --muted:#9fb0d0;
        --accent:#7c5cff;
        --danger:#ff4d6d;
        --chip:#1b2a4d;
        --chipActive:#2b46a3;
        --border:rgba(255,255,255,.10);
        --shadow: 0 12px 30px rgba(0,0,0,.35);
        --radius:16px;
      }
      body{
        margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans KR", Arial;
        background: radial-gradient(1200px 600px at 20% 0%, rgba(124,92,255,.30), transparent 55%),
                    radial-gradient(900px 500px at 80% 10%, rgba(0,212,255,.18), transparent 50%),
                    var(--bg);
        color:var(--text);
      }
      .wrap{max-width:1100px; margin:0 auto; padding:22px;}
      .header{
        display:flex; align-items:center; gap:14px;
        padding:14px 14px; border:1px solid var(--border);
        background: linear-gradient(180deg, rgba(124,92,255,.18), rgba(18,28,51,.0));
        border-radius:var(--radius); box-shadow:var(--shadow);
      }
      .logo{display:flex; align-items:center; gap:12px; flex:1;}
      .logo img{height:42px; width:auto;}
      .titlebox{display:flex; flex-direction:column; line-height:1.1;}
      .titlebox .t1{font-weight:800; font-size:18px;}
      .titlebox .t2{color:var(--muted); font-size:12px; margin-top:4px;}
      .hero{
        margin-top:14px;
        width:100%; border:1px solid var(--border);
        border-radius:var(--radius); overflow:hidden;
        background: var(--card2);
      }
      .hero img{width:100%; height:120px; object-fit:cover; display:block;}
      .grid{
        display:grid; grid-template-columns: 1fr; gap:14px; margin-top:14px;
      }
      @media(min-width:980px){
        .grid{grid-template-columns: 1fr 360px;}
      }
      details{
        border:1px solid var(--border);
        background: rgba(18,28,51,.65);
        border-radius:var(--radius);
        padding:12px 12px; box-shadow: 0 6px 20px rgba(0,0,0,.22);
      }
      summary{
        cursor:pointer; font-weight:800;
        display:flex; align-items:center; gap:10px;
      }
      summary .sumicon{
        width:32px; height:32px; display:inline-flex; align-items:center; justify-content:center;
        border-radius:12px; background: rgba(124,92,255,.22); border:1px solid rgba(124,92,255,.35);
      }
      .row{display:flex; gap:10px; flex-wrap:wrap; align-items:center;}
      label{color:var(--muted); font-size:12px;}
      input[type="range"]{width:100%;}
      .pill{
        border:1px solid var(--border);
        background: var(--chip);
        border-radius:999px; padding:8px 10px;
        color:var(--text); font-size:13px;
        cursor:pointer; user-select:none;
      }
      .pill.active{background: var(--chipActive); border-color: rgba(124,92,255,.55);}
      .pill.danger{border-color: rgba(255,77,109,.50); background: rgba(255,77,109,.10);}
      .cardlist{display:grid; grid-template-columns: 1fr; gap:10px;}
      @media(min-width:680px){
        .cardlist{grid-template-columns: 1fr 1fr;}
      }
      .result-card{
        border:1px solid var(--border);
        background: rgba(18,28,51,.72);
        border-radius:var(--radius);
        padding:12px;
      }
      .card-top{
        display:flex; align-items:center; gap:10px; margin-bottom:8px;
      }
      .iconbox{
        width:38px; height:38px; display:flex; align-items:center; justify-content:center;
        border-radius:14px; background: rgba(124,92,255,.18);
        border: 1px solid rgba(124,92,255,.35);
      }
      .iconbox img{width:22px; height:22px;}
      .result-title{font-weight:900;}
      .muted{color:var(--muted); font-size:12px; line-height:1.4;}
      .bigmsg{font-weight:800; font-size:14px; margin-top:6px;}
      textarea, input[type="number"], input[type="text"]{
        width:100%;
        background: rgba(15,24,48,.65);
        color:var(--text);
        border:1px solid var(--border);
        border-radius:14px;
        padding:10px 12px;
        outline:none;
      }
      textarea{min-height:74px; resize:vertical;}
      .btn{
        width:100%; padding:12px 14px;
        border-radius:14px; border: 1px solid rgba(124,92,255,.65);
        background: rgba(124,92,255,.22);
        color:var(--text);
        font-weight:900; cursor:pointer;
      }
      .btn.primary{
        background: linear-gradient(90deg, rgba(124,92,255,.45), rgba(0,212,255,.18));
        border-color: rgba(124,92,255,.85);
      }
      .btn:disabled{opacity:.6; cursor:not-allowed;}
      .whyBtn{
        width:auto !important;
        padding:10px 12px !important;
        margin-top:10px;
        display:inline-flex;
        justify-content:center;
        border-radius:12px;
      }
      .expDetails details{
        margin-top:10px;
      }
      .expDetails summary{
        cursor:pointer;
        color:var(--muted);
        font-weight:900;
        outline:none;
      }
      .section-note{margin-top:8px; color:var(--muted); font-size:12px;}
      .chatbox{
        border:1px solid var(--border);
        background: rgba(18,28,51,.72);
        border-radius:var(--radius);
        padding:12px;
        min-height:260px;
        display:flex; flex-direction:column;
        gap:10px;
      }
      .chatmsgs{
        flex:1; overflow:auto;
        padding-right:6px;
        display:flex; flex-direction:column; gap:10px;
      }
      .bubble{
        padding:10px 12px; border-radius:14px; border:1px solid var(--border);
        max-width: 100%;
      }
      .bubble.user{background: rgba(0,212,255,.12); border-color: rgba(0,212,255,.22);}
      .bubble.assistant{background: rgba(124,92,255,.14); border-color: rgba(124,92,255,.26);}
      .chatinputrow{display:flex; gap:10px;}
      .chatinputrow input{flex:1;}
      .hr{height:1px; background: var(--border); margin:10px 0;}
      .assetimg img{height:120px; display:block;}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="header">
        <div class="logo">
          <img id="logoImg" src="/assets/logo_babycoach.svg" alt="BabyCoach 로고"
               onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=&quot;http://www.w3.org/2000/svg&quot; width=&quot;240&quot; height=&quot;80&quot; viewBox=&quot;0 0 240 80&quot;><rect width=&quot;240&quot; height=&quot;80&quot; rx=&quot;16&quot; fill=&quot;%237c5cff&quot;/><text x=&quot;50%25&quot; y=&quot;55%25&quot; dominant-baseline=&quot;middle&quot; text-anchor=&quot;middle&quot; font-family=&quot;Arial&quot; font-size=&quot;22&quot; fill=&quot;white&quot; font-weight=&quot;700&quot;>BabyCoach</text></svg>';" />
          <div class="titlebox">
            <div class="t1">BabyCoach PoC</div>
            <div class="t2">LangGraph 추천 + 단일 챗봇</div>
          </div>
        </div>
      </div>

      <div class="hero">
        <img id="heroImg" src="/assets/hero_babycoach.svg" alt="Hero 배너"
             onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=&quot;http://www.w3.org/2000/svg&quot; width=&quot;1200&quot; height=&quot;120&quot; viewBox=&quot;0 0 1200 120&quot;><defs><linearGradient id=&quot;g&quot; x1=&quot;0&quot; y1=&quot;0&quot; x2=&quot;1&quot; y2=&quot;0&quot;><stop offset=&quot;0&quot; stop-color=&quot;%237c5cff&quot; stop-opacity=&quot;0.7&quot;/><stop offset=&quot;1&quot; stop-color=&quot;%2300d4ff&quot; stop-opacity=&quot;0.25&quot;/></linearGradient></defs><rect width=&quot;1200&quot; height=&quot;120&quot; fill=&quot;url(%23g)&quot;/><text x=&quot;60&quot; y=&quot;70&quot; font-family=&quot;Arial&quot; font-size=&quot;30&quot; fill=&quot;white&quot; font-weight=&quot;700&quot;>오늘의 상태 체크, 가볍게 추천</text></svg>';" />
      </div>

      <div class="grid">
        <div style="display:flex; flex-direction:column; gap:14px;">
          <details open>
            <summary>
              <span class="sumicon">
                <img src="/assets/icon_profile.svg" alt="Profile"
                     onerror="this.onerror=null; this.outerHTML='<span class=&quot;sumicon&quot;>👶</span>';" style="width:18px; height:18px;"/>
              </span>
              아기 기본 정보
            </summary>

            <div style="margin-top:10px;">
              <div class="row" style="width:100%;">
                <div style="flex:1; min-width:220px;">
                  <label>월령 (0~36개월)</label>
                  <div class="bigmsg" id="ageMonthsLabel">0개월</div>
                  <input id="age_months" type="range" min="0" max="36" value="10" step="1"
                         oninput="ui.setAgeMonths(this.value)" />
                </div>
                <div style="flex:1; min-width:220px;">
                  <label>체중 (kg)</label>
                  <input id="weight_kg" type="number" min="0" max="50" value="8.7" step="0.1" placeholder="예: 8.7" />
                </div>
              </div>

              <div class="section-note">알레르기는 태그 형태로 선택하고, “없음”도 지원해요.</div>
              <div class="row" style="margin-top:10px;">
                <div class="pill" data-chip="allergy" data-value="없음" onclick="ui.toggleChip(this,'allergies','없음')">없음</div>
                <div class="pill" data-chip="allergy" data-value="달걀흰자" onclick="ui.toggleChip(this,'allergies','달걀흰자')">달걀흰자</div>
                <div class="pill" data-chip="allergy" data-value="우유" onclick="ui.toggleChip(this,'allergies','우유')">우유</div>
                <div class="pill" data-chip="allergy" data-value="대두" onclick="ui.toggleChip(this,'allergies','대두')">대두</div>
                <div class="pill" data-chip="allergy" data-value="견과류" onclick="ui.toggleChip(this,'allergies','견과류')">견과류</div>
              </div>
              <div class="row" style="margin-top:8px;">
                <label style="margin-right:8px;">추가 알레르기</label>
                <input id="allergy_custom" type="text" placeholder="직접 입력 (예: 새우)" style="flex:1; min-width:220px;" />
                <button class="btn" style="width:auto; padding:10px 12px;" onclick="ui.addCustomAllergy()">추가</button>
              </div>
              <div style="margin-top:10px;">
                <label>메모</label>
                <textarea id="notes" placeholder="예: 새로운 음식은 조심스럽게 도입 중"></textarea>
              </div>
            </div>
          </details>

          <details open>
            <summary>
              <span class="sumicon">
                <img src="/assets/icon_meal.svg" alt="Meal"
                     onerror="this.onerror=null; this.outerHTML='<span class=&quot;sumicon&quot;>🍲</span>';" style="width:18px; height:18px;"/>
              </span>
              이유식 / 식사 상태
            </summary>
            <div style="margin-top:10px;">
              <div class="row" style="margin-bottom:6px;">
                <div style="flex:1; min-width:240px;">
                  <label>단백질 단계 (0~3)</label>
                  <div class="row" id="protein_stage" style="margin-top:8px;">
                    <div class="pill" onclick="ui.setStage(this,'protein_count_3d',0)">0회</div>
                    <div class="pill" onclick="ui.setStage(this,'protein_count_3d',1)">1회</div>
                    <div class="pill" onclick="ui.setStage(this,'protein_count_3d',2)">2회</div>
                    <div class="pill" onclick="ui.setStage(this,'protein_count_3d',3)">3회 이상</div>
                  </div>
                  <div class="section-note" id="proteinStageHint">현재: 2</div>
                </div>
                <div style="flex:1; min-width:240px;">
                  <label>채소 단계 (0~3)</label>
                  <div class="row" id="vegetable_stage" style="margin-top:8px;">
                    <div class="pill" onclick="ui.setStage(this,'vegetable_count_3d',0)">0회</div>
                    <div class="pill" onclick="ui.setStage(this,'vegetable_count_3d',1)">1회</div>
                    <div class="pill" onclick="ui.setStage(this,'vegetable_count_3d',2)">2회</div>
                    <div class="pill" onclick="ui.setStage(this,'vegetable_count_3d',3)">3회 이상</div>
                  </div>
                  <div class="section-note" id="vegetableStageHint">현재: 1</div>
                </div>
              </div>

              <div class="row" style="margin-top:12px;">
                <div style="flex:1; min-width:240px;">
                  <label>영양 다양성 (1~10)</label>
                  <div class="bigmsg" id="diversityLabel">6점</div>
                  <input id="food_diversity_3d" type="range" min="1" max="10" value="6" step="1"
                         oninput="ui.setDiversity(this.value)" />
                  <div class="section-note">툴팁: 여러 식재료를 조금씩 접해본 정도예요.</div>
                </div>
              </div>

              <div class="row" style="margin-top:12px;">
                <div style="flex:1; min-width:240px;">
                  <label>식사 태그</label>
                  <div class="bigmsg" id="foodTagLabel">다양성 중심</div>
                  <div class="row" style="margin-top:8px;">
                    <div class="pill" data-singlekey="food_tag" onclick="ui.selectSinglePill(this,'food_tag','단백질 중심','foodTagLabel','단백질 중심')">단백질 중심</div>
                    <div class="pill" data-singlekey="food_tag" onclick="ui.selectSinglePill(this,'food_tag','채소 중심','foodTagLabel','채소 중심')">채소 중심</div>
                    <div class="pill active" data-singlekey="food_tag" onclick="ui.selectSinglePill(this,'food_tag','다양성 중심','foodTagLabel','다양성 중심')">다양성 중심</div>
                  </div>
                </div>
                <div style="flex:1; min-width:240px;">
                  <label>식사 반응</label>
                  <div class="bigmsg" id="mealReactionLabel">괜찮아요</div>
                  <div class="row" style="margin-top:8px;">
                    <div class="pill active" data-singlekey="meal_reaction" onclick="ui.selectSinglePill(this,'meal_reaction','괜찮아요','mealReactionLabel','괜찮아요')">괜찮아요</div>
                    <div class="pill" data-singlekey="meal_reaction" onclick="ui.selectSinglePill(this,'meal_reaction','조심스러워요','mealReactionLabel','조심스러워요')">조심스러워요</div>
                    <div class="pill" data-singlekey="meal_reaction" onclick="ui.selectSinglePill(this,'meal_reaction','거부 신호','mealReactionLabel','거부 신호')">거부 신호</div>
                  </div>
                </div>
              </div>

              <div class="row" style="margin-top:10px;">
                <div class="pill danger" id="meal_refusal_pill" onclick="ui.toggleFlag('meal_refusal',this)">식사 거부가 있었어요</div>
                <div style="flex:1;"></div>
                <div style="flex:1;"></div>
              </div>

              <div style="margin-top:12px;">
                <label>반응 플래그 (다중 선택)</label>
                <div class="row" style="margin-top:8px;">
                  <div class="pill" onclick="ui.toggleFlagChip(this,'reaction_flags','없음')">없음</div>
                  <div class="pill" onclick="ui.toggleFlagChip(this,'reaction_flags','발진')">발진</div>
                  <div class="pill" onclick="ui.toggleFlagChip(this,'reaction_flags','구토')">구토</div>
                  <div class="pill" onclick="ui.toggleFlagChip(this,'reaction_flags','설사')">설사</div>
                  <div class="pill" onclick="ui.toggleFlagChip(this,'reaction_flags','변비')">변비</div>
                </div>
              </div>
            </div>
          </details>

          <details open>
            <summary>
              <span class="sumicon">
                <img src="/assets/icon_play.svg" alt="Play"
                     onerror="this.onerror=null; this.outerHTML='<span class=&quot;sumicon&quot;>🧸</span>';" style="width:18px; height:18px;"/>
              </span>
              놀이 상태
            </summary>
            <div style="margin-top:10px;">
              <div style="margin-bottom:8px;">
                <label>놀이 유형 (다중 선택)</label>
                <div class="row" style="margin-top:8px;">
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','촉감 놀이')">촉감 놀이</div>
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','딸랑이 흔들기')">딸랑이 흔들기</div>
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','넣기/빼기')">넣기/빼기</div>
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','쌓기 놀이')">쌓기 놀이</div>
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','버튼 누르기')">버튼 누르기</div>
                  <div class="pill" onclick="ui.toggleChip(this,'play_types','시각 추적')">시각 추적</div>
                </div>
              </div>

              <div class="row">
                <div style="flex:1; min-width:240px;">
                  <label>놀이 집중 레벨</label>
                  <div class="bigmsg" id="focusLevelLabel">집중: 중간</div>
                  <div class="row" style="margin-top:8px;">
                    <div class="pill" data-focuslevel="낮음" onclick="ui.setFocusLevel('낮음')">낮음</div>
                    <div class="pill" data-focuslevel="중간" onclick="ui.setFocusLevel('중간')">중간</div>
                    <div class="pill" data-focuslevel="높음" onclick="ui.setFocusLevel('높음')">높음</div>
                  </div>
                  <input id="focus_minutes" type="hidden" value="10" />
                </div>
                <div style="flex:1; min-width:240px;">
                  <label>반복 시도 횟수 (0~10)</label>
                  <div class="bigmsg" id="repeatLabel">반복 시도: 3</div>
                  <input id="repeat_count" type="range" min="0" max="10" value="3" step="1"
                         oninput="ui.setRepeat(this.value)" />
                </div>
              </div>

              <div style="margin-top:12px;">
                <label>아이 주도 비율 (0.0~1.0)</label>
                <div class="bigmsg" id="ledRatioLabel">아이 주도 비율: 0.40 (중간)</div>
                <input id="child_led_ratio" type="range" min="0" max="1" value="0.4" step="0.01"
                       oninput="ui.setLedRatio(this.value)" />
                <div class="section-note">낮음: 부모 주도 / 중간 / 높음: 아이 주도</div>
              </div>

              <div style="margin-top:10px;">
                <div class="pill danger" onclick="ui.toggleFlag('refusal',this)">놀이를 거부했어요</div>
              </div>

              <div style="margin-top:10px;">
                <label>부모 메모</label>
                <input id="parent_note" type="text" placeholder="예: 손으로 만지고 흔드는 놀이를 좋아함" />
              </div>
            </div>
          </details>

          <details open>
            <summary>
              <span class="sumicon">
                <img src="/assets/icon_interaction.svg" alt="Interaction"
                     onerror="this.onerror=null; this.outerHTML='<span class=&quot;sumicon&quot;>🤝</span>';" style="width:18px; height:18px;"/>
              </span>
              부모-아기 상호작용
            </summary>
            <div style="margin-top:10px;">
              <div class="row">
                <div style="flex:1; min-width:240px;">
                  <label>스킨십/터치 빈도 (0~10)</label>
                  <input id="touch_count" type="range" min="0" max="10" value="3" step="1" oninput="ui.setSimpleSlider('touch_count',this.value,'touchLabel','터치 빈도: ')"/>
                  <div class="section-note" id="touchLabel">터치 빈도: 3</div>
                </div>
                <div style="flex:1; min-width:240px;">
                  <label>말 걸기/이름 붙여주기 빈도 (0~10)</label>
                  <input id="labeling_count" type="range" min="0" max="10" value="2" step="1" oninput="ui.setSimpleSlider('labeling_count',this.value,'labelingLabel','말 걸기: ')"/>
                  <div class="section-note" id="labelingLabel">말 걸기: 2</div>
                </div>
              </div>

              <div class="row" style="margin-top:10px;">
                <div style="flex:1; min-width:240px;">
                  <label>같이 보고 반응한 횟수 (0~10)</label>
                  <input id="joint_attention_count" type="range" min="0" max="10" value="4" step="1" oninput="ui.setSimpleSlider('joint_attention_count',this.value,'jointLabel','같이 보기: ')"/>
                  <div class="section-note" id="jointLabel">같이 보기: 4</div>
                </div>
                <div style="flex:1; min-width:240px;">
                  <label>아기 반응에 맞춘 주고받기 (0~10)</label>
                  <input id="responsive_turns" type="range" min="0" max="10" value="2" step="1" oninput="ui.setSimpleSlider('responsive_turns',this.value,'responsiveLabel','주고받기: ')"/>
                  <div class="section-note" id="responsiveLabel">주고받기: 2</div>
                </div>
              </div>

              <div style="margin-top:10px;">
                <div class="pill danger" onclick="ui.toggleFlag('flat_response',this)">오늘은 반응이 전반적으로 적었어요</div>
              </div>
            </div>
          </details>

          <details open>
            <summary>
              <span class="sumicon">
                <img src="/assets/chatbot_avatar.svg" alt="Chat"
                     onerror="this.onerror=null; this.outerHTML='<span class=&quot;sumicon&quot;>💬</span>';" style="width:18px; height:18px;"/>
              </span>
              궁금한 점
            </summary>
            <div style="margin-top:10px;">
              <textarea id="parent_query" placeholder="오늘은 어떤 놀이와 관찰 포인트가 좋을까요?"></textarea>
              <div class="row" style="margin-top:10px;">
                <button class="pill" style="border-radius:14px;" onclick="ui.setQuery('오늘 놀이 추천 받기')">오늘 놀이 추천 받기</button>
                <button class="pill" style="border-radius:14px;" onclick="ui.setQuery('식사 조언 받기')">식사 조언 받기</button>
                <button class="pill" style="border-radius:14px;" onclick="ui.setQuery('관찰 포인트 보기')">관찰 포인트 보기</button>
                <button class="pill" style="border-radius:14px;" onclick="ui.setQuery('상호작용 팁 받기')">상호작용 팁 받기</button>
              </div>
            </div>
          </details>

          <button class="btn primary" id="recommendBtn" onclick="ui.submitRecommend()">오늘의 BabyCoach 추천 받기</button>
          <div class="section-note" id="statusText"></div>
        </div>

        <div class="chatbox">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
            <div style="font-weight:900;">BabyCoach Chat</div>
            <div class="muted" style="font-size:12px;">DB 없음 · 추천 기반</div>
          </div>
          <div class="chatmsgs" id="chatMsgs"></div>
          <div class="hr"></div>
          <div class="chatinputrow">
            <input id="chatUserMessage" type="text" placeholder="질문을 입력해 주세요 (예: 왜 이런 놀이를 추천했어?)" />
            <button class="btn" style="width:auto; padding:10px 12px;" onclick="ui.submitChat()">전송</button>
          </div>
        </div>
      </div>

      <div style="margin-top:14px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
          <div class="iconbox" style="width:44px; height:44px;">
            <img src="/assets/empty_state_result.svg" alt="Result"
                 onerror="this.onerror=null; this.outerHTML='<span style=&quot;color:white;font-weight:900;&quot;>✨</span>';" />
          </div>
          <div style="font-weight:900;">결과 카드</div>
          <div class="muted">추천 받기 버튼을 누르면 아래에 표시돼요.</div>
        </div>
        <div class="cardlist" id="resultCards"></div>
      </div>
    </div>

    <script>
      const ui = {
        payload: {
          age_months: 10,
          weight_kg: 8.7,
          allergies: [],
          notes: "",
          protein_count_3d: 2,
          vegetable_count_3d: 1,
          food_diversity_3d: 6,
          meal_refusal: false,
          reaction_flags: [],
          food_tag: "다양성 중심",
          meal_reaction: "괜찮아요",
          play_types: [],
          play_focus_level: "중간",
          focus_minutes: 10,
          repeat_count: 3,
          child_led_ratio: 0.4,
          refusal: false,
          parent_note: "",
          touch_count: 3,
          labeling_count: 2,
          joint_attention_count: 4,
          responsive_turns: 2,
          flat_response: false,
          parent_query: ""
        },
        _chipActive(el){
          el.classList.add('active');
        },
        _chipInactive(el){
          el.classList.remove('active');
        },
        setAgeMonths(v){
          this.payload.age_months = Number(v);
          document.getElementById('ageMonthsLabel').textContent = `${v}개월`;
        },
        setDiversity(v){
          this.payload.food_diversity_3d = Number(v);
          document.getElementById('diversityLabel').textContent = `${v}점`;
        },
        setFocus(v){
          this.payload.focus_minutes = Number(v);
          const el2 = document.getElementById('focusLabel');
          if (el2) el2.textContent = `집중 시간: ${v}분`;
          const el1 = document.getElementById('focusLevelLabel');
          if (el1){
            const lvl = this.payload.play_focus_level || '';
            el1.textContent = lvl ? `집중: ${lvl}` : `집중: ${v}분`;
          }
        },
        setFocusLevel(level){
          let minutes = 10;
          if (level === '낮음') minutes = 3;
          else if (level === '중간') minutes = 10;
          else minutes = 20;

          this.payload.play_focus_level = level;
          this.payload.focus_minutes = minutes;

          const hidden = document.getElementById('focus_minutes');
          if (hidden) hidden.value = String(minutes);

          const labelEl = document.getElementById('focusLevelLabel');
          if (labelEl) labelEl.textContent = `집중: ${level}`;

          // Mark active pill for focus level.
          document.querySelectorAll('[data-focuslevel]').forEach(e => e.classList.remove('active'));
          const activePill = document.querySelector(`[data-focuslevel="${level}"]`);
          if (activePill) activePill.classList.add('active');
        },
        setRepeat(v){
          this.payload.repeat_count = Number(v);
          document.getElementById('repeatLabel').textContent = `반복 시도: ${v}`;
        },
        setLedRatio(v){
          const r = Number(v);
          this.payload.child_led_ratio = r;
          let label = '중간';
          if (r <= 0.35) label = '부모 주도';
          else if (r >= 0.7) label = '아이 주도';
          document.getElementById('ledRatioLabel').textContent = `아이 주도 비율: ${r.toFixed(2)} (${label})`;
        },
        setStage(el, key, v){
          this.payload[key] = Number(v);
          // Update hint.
          const hintId = key === 'protein_count_3d' ? 'proteinStageHint' : 'vegetableStageHint';
          document.getElementById(hintId).textContent = `현재: ${v}`;

          // mark active
          const parent = el.parentElement;
          [...parent.children].forEach(c => c.classList.remove('active'));
          el.classList.add('active');
        },
        setSimpleSlider(key, v, labelId, prefix){
          this.payload[key] = Number(v);
          document.getElementById(labelId).textContent = prefix + v;
        },
        toggleFlag(key, el){
          this.payload[key] = !this.payload[key];
          el.classList.toggle('active', this.payload[key]);

          if (key === 'meal_refusal'){
            // Keep `meal_reaction` consistent with `meal_refusal`.
            const mealReaction = this.payload.meal_refusal ? '거부 신호' : '괜찮아요';
            this.payload.meal_reaction = mealReaction;
            const labelEl = document.getElementById('mealReactionLabel');
            if (labelEl) labelEl.textContent = mealReaction;

            // Clear reaction flags UI + payload for consistency.
            this.payload.reaction_flags = [];
            document.querySelectorAll("[onclick*='reaction_flags']").forEach(e => e.classList.remove('active'));

            // Update meal_reaction active pill.
            document.querySelectorAll('[data-singlekey=\"meal_reaction\"]').forEach(e => e.classList.remove('active'));
            const match = [...document.querySelectorAll('[data-singlekey=\"meal_reaction\"]')].find(e => (e.textContent || '').trim() === mealReaction);
            if (match) match.classList.add('active');
          }
        },
        selectSinglePill(el, key, value, labelId, labelText){
          this.payload[key] = value;

          // Active state for this single-selection group.
          document.querySelectorAll(`[data-singlekey="${key}"]`).forEach(e => e.classList.remove('active'));
          el.classList.add('active');

          const labelEl = document.getElementById(labelId);
          if (labelEl) labelEl.textContent = labelText;

          // Synchronize meal_reaction with the existing meal_refusal + reaction_flags UI.
          if (key === 'meal_reaction'){
            const isRefusal = value === '거부 신호';
            this.payload.meal_refusal = isRefusal;

            const mealRefusalPill = document.getElementById('meal_refusal_pill');
            if (mealRefusalPill) mealRefusalPill.classList.toggle('active', isRefusal);

            // Clear symptom flags when meal_reaction changes.
            this.payload.reaction_flags = [];
            document.querySelectorAll("[onclick*='reaction_flags']").forEach(e => e.classList.remove('active'));
          }
        },
        toggleChip(el, key, value){
          if (!this.payload[key]) this.payload[key] = [];

          // Special "없음" behavior for allergies
          if (key === 'allergies' && value === '없음'){
            this.payload.allergies = [];
            // reset chip visuals
            document.querySelectorAll('[data-chip="allergy"]').forEach(e => e.classList.remove('active'));
            el.classList.add('active');
            return;
          }

          // If any allergy added, remove "없음" state.
          if (key === 'allergies'){
            this.payload.allergies = this.payload.allergies.filter(x => x !== '없음');
          }

          const idx = this.payload[key].indexOf(value);
          if (idx >= 0){
            this.payload[key].splice(idx, 1);
            el.classList.remove('active');
          }else{
            this.payload[key].push(value);
            el.classList.add('active');
            if (key === 'allergies'){
              const noneChip = document.querySelector('[data-chip="allergy"][data-value="없음"]');
              if (noneChip) noneChip.classList.remove('active');
            }
          }
        },
        toggleFlagChip(el, key, value){
          if (!this.payload[key]) this.payload[key] = [];
          if (value === '없음'){
            this.payload[key] = [];
            document.querySelectorAll(`[onclick*=&quot;${key}&quot;]`).forEach(x => x.classList.remove('active'));
            el.classList.add('active');
            return;
          }
          // remove '없음' chip if present
          this.payload[key] = this.payload[key].filter(v => v !== '없음');

          const idx = this.payload[key].indexOf(value);
          if (idx >= 0){
            this.payload[key].splice(idx,1);
            el.classList.remove('active');
          }else{
            this.payload[key].push(value);
            el.classList.add('active');
            // unselect '없음'
            const noneChip = document.querySelector(`div[onclick*="없음"]`);
            if (noneChip) noneChip.classList.remove('active');
          }
        },
        addCustomAllergy(){
          const v = (document.getElementById('allergy_custom').value || '').trim();
          if (!v) return;
          if (!this.payload.allergies) this.payload.allergies = [];
          if (v === '없음'){
            this.payload.allergies = [];
            document.querySelectorAll('[data-chip="allergy"]').forEach(e => e.classList.remove('active'));
            const noneChip = document.querySelector('[data-chip="allergy"][data-value="없음"]');
            if (noneChip) noneChip.classList.add('active');
            return;
          }
          this.payload.allergies.push(v);
          document.getElementById('allergy_custom').value = '';
          alert('추가 알레르기: ' + v);
        },
        setQuery(text){
          const textarea = document.getElementById('parent_query');
          const presetMap = {
            '오늘 놀이 추천 받기': '오늘은 어떤 놀이와 관찰 포인트가 좋을까요?',
            '식사 조언 받기': '오늘 식사에서 어떤 재료/순서가 도움이 될까요?',
            '관찰 포인트 보기': '오늘 관찰하면 좋은 포인트를 알려줘.',
            '상호작용 팁 받기': '상호작용을 늘리려면 어떻게 해야 해?'
          };
          textarea.value = presetMap[text] || text;
          this.payload.parent_query = textarea.value;
        },
        submitRecommend: async function(){
          const btn = document.getElementById('recommendBtn');
          btn.disabled = true;
          document.getElementById('statusText').textContent = '추천 생성 중...';

          // sync text inputs
          this.payload.weight_kg = Number(document.getElementById('weight_kg').value);
          this.payload.notes = document.getElementById('notes').value || '';
          this.payload.refusal = !!this.payload.refusal;
          this.payload.parent_note = document.getElementById('parent_note').value || '';
          this.payload.parent_query = document.getElementById('parent_query').value || '';

          const res = await fetch('/recommend', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(this.payload)
          });
          const data = await res.json();
          if (!res.ok){
            document.getElementById('statusText').textContent = '오류: ' + (data.detail || 'unknown');
            btn.disabled = false;
            return;
          }
          this.final_output = data.final_output;
          document.getElementById('statusText').textContent = '완료! 결과를 확인해 주세요.';
          this.renderResults(this.final_output);
          this.resetChat(data.final_output);
          btn.disabled = false;
        },
        renderIcon(imgPath, fallbackText){
          // Return HTML for icon with graceful fallback.
          return `
            <img src="${imgPath}" alt="" onerror="this.onerror=null; this.outerHTML='<span style=&quot;font-weight:900;&quot;>${fallbackText}</span>';" />
          `;
        },
        renderResults: function(final_output){
          const el = document.getElementById('resultCards');
          el.innerHTML = '';

          const spoon = final_output.spoon || {};
          const play = final_output.play || {};
          const growth = final_output.growth || {};
          const nudge = final_output.nudge || {};
          const explanation = final_output.explanation || {};

          const cards = [
            { key:'spoon', title:'Spoon', icon:'/assets/icon_spoon.svg', fallback:'🥄', body: this.renderList(spoon.suggestions || [], spoon.notes || '') },
            { key:'play', title:'Play', icon:'/assets/icon_play.svg', fallback:'🧸', body: this.renderList(play.suggestions || [], play.notes || '') },
            { key:'growth', title:'Growth', icon:'/assets/icon_growth.svg', fallback:'📈', body: this.renderList(growth.observation_points || [], '') },
            { key:'nudge', title:'오늘의 한 문장 코칭', icon:'/assets/icon_growth.svg', fallback:'✨', body: `
              <div class="bigmsg">${escapeHtml(nudge.nudge_message || '')}</div>
              <button class="btn whyBtn" style="width:auto;" onclick="ui.askWhy()">왜?</button>
            ` },
            { key:'explanation', title:'설명', icon:'/assets/icon_growth.svg', fallback:'🧠', body: `
              <div class="expDetails">
                <details>
                  <summary>설명 펼치기/접기</summary>
                  <div class="muted" style="margin-top:8px;">${escapeHtml(explanation.explanation || '')}</div>
                </details>
              </div>
            ` }
          ];

          cards.forEach((c) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
              <div class="card-top">
                <div class="iconbox">${this.renderIcon(c.icon, c.fallback)}</div>
                <div class="result-title">${c.title}</div>
              </div>
              ${c.body}
            `;
            el.appendChild(card);
          });
        },
        renderList: function(items, notes){
          let html = '';
          if (items && items.length){
            html += '<ul style="margin:0; padding-left:18px; color:var(--text);">';
            items.forEach(it => { html += `<li style="margin:6px 0;">${escapeHtml(it)}</li>`; });
            html += '</ul>';
          }
          if (notes){
            html += `<div class="muted" style="margin-top:8px;">${escapeHtml(notes)}</div>`;
          }
          return html || `<div class="muted">-</div>`;
        },
        resetChat: function(final_output){
          const msgs = document.getElementById('chatMsgs');
          msgs.innerHTML = '';
          const first = "추천 결과가 나왔어요. 궁금한 점을 물어봐 주세요. 예: “왜 이런 놀이를 추천했어?”";
          msgs.appendChild(this.bubble('assistant', first));
          document.getElementById('chatUserMessage').value = '';
        },
        bubble: function(role, text){
          const div = document.createElement('div');
          div.className = 'bubble ' + (role === 'user' ? 'user' : 'assistant');
          div.textContent = text;
          return div;
        },
        askWhy: async function(){
          // Preset question that uses the current `final_output` context.
          const preset = "왜 이런 추천이 나왔어?";
          document.getElementById('chatUserMessage').value = preset;
          return this.submitChat();
        },
        submitChat: async function(){
          if (!this.final_output){
            alert('먼저 추천을 받아주세요.');
            return;
          }
          const input = document.getElementById('chatUserMessage');
          const msg = (input.value || '').trim();
          if (!msg) return;

          const msgs = document.getElementById('chatMsgs');
          msgs.appendChild(this.bubble('user', msg));
          msgs.scrollTop = msgs.scrollHeight;
          input.value = '';

          const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
              final_output: this.final_output,
              state_summary: this.final_output.chat_context_summary,
              user_message: msg
            })
          });
          const data = await res.json();
          const assistant = (data.assistant_message || ('오류: ' + (data.detail || 'unknown')) );
          msgs.appendChild(this.bubble('assistant', assistant));
          msgs.scrollTop = msgs.scrollHeight;
        }
      };

      function escapeHtml(s){
        return String(s)
          .replaceAll('&','&amp;')
          .replaceAll('<','&lt;')
          .replaceAll('>','&gt;')
          .replaceAll('"','&quot;')
          .replaceAll(\"'\",'&#039;');
      }

      // Initialize stage labels and some active states.
      window.addEventListener('load', () => {
        ui.setAgeMonths(document.getElementById('age_months').value);
        ui.setDiversity(document.getElementById('food_diversity_3d').value);
        ui.setFocusLevel(ui.payload.play_focus_level || '중간');
        ui.setRepeat(document.getElementById('repeat_count').value);
        ui.setLedRatio(document.getElementById('child_led_ratio').value);

        // Set initial active chips for the first stage selections.
        // (We don't force all chip states here to keep it simple.)
        // Bind initial click state for stage buttons:
        const proteinButtons = document.getElementById('protein_stage').children;
        [...proteinButtons].forEach(btn => {
          const label = btn.textContent;
          if (label.includes('2')) btn.classList.add('active');
        });
        const vegButtons = document.getElementById('vegetable_stage').children;
        [...vegButtons].forEach(btn => {
          const label = btn.textContent;
          if (label.includes('1')) btn.classList.add('active');
        });
      });
    </script>
  </body>
</html>
"""

