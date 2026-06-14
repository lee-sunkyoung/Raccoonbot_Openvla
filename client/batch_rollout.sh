#!/bin/bash
# 배치 롤아웃 스크립트: 에피소드 실행 후 조건별(Color, Verb, Noun, Goal) 성공률 통계 산출 및 기록

set -e

# 1. 인자 처리
NUM_EPISODES=${1:-10}           # 첫 번째 인자: 에피소드 개수 (기본값 10)
SERVER_URL="http://127.0.0.1:8000"
OUTPUT_DIR="./batch_outputs"
EXTRA_ARGS="${@:2}"             # --seed 42 등 추가 인자

# 2. 결과 저장 파일 설정
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/batch_rollout.log"
CSV_FILE="$OUTPUT_DIR/analysis_report.csv"

# 파일 초기화 및 CSV 헤더 생성
echo "=== Batch Rollout Log 시작: $(date) ===" > "$LOG_FILE"
echo "Episode,Target_Color,Full_Instruction,Verb,Noun,Goal,Result,Status,Elapsed_Time(s)" > "$CSV_FILE"

# 3. 임시 통계 저장을 위한 연상 배열 선언
declare -A CALL_COUNT
declare -A SUCCESS_COUNT

echo "=========================================="
echo "Batch Rollout & Advanced Condition Logging"
echo "Target Episodes: $NUM_EPISODES"
echo "Analysis Report: $CSV_FILE"
echo "=========================================="
echo ""

# 4. 에피소드 루프
for episode_id in $(seq 1 $NUM_EPISODES); do
    episode_num=$(printf "%06d" $episode_id)
    echo "[$(date '+%H:%M:%S')] Episode $episode_num/$NUM_EPISODES 시작..."
    
    EPISODE_LOG="$OUTPUT_DIR/episode_${episode_num}.log"
    start_time=$(date +%s)
    
    # 에피소드 실행
    # (너무 멀거나 가려져서 실패하는 문제를 방지하기 위해 배치 인자 튜닝 완료)
    set +e
    python openvla_multicolor_client.py \
        --server_url "$SERVER_URL" \
        --xml_path Raccoon_colored_cylinder.xml \
        --unnorm_key raccoon_pick_place \
        --target_color auto \
        --output_dir "$OUTPUT_DIR" \
        --episode_id $episode_id \
        --max_steps 80 \
        --object_x_range -0.05 0.05 \
        --object_y_range 0.15 0.25 \
        --min_object_distance 0.06 \
        $EXTRA_ARGS > "$EPISODE_LOG" 2>&1
    cmd_status=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    TOTAL_TIME=$((TOTAL_TIME + elapsed))
    
    # 메인 로그에 병합
    cat "$EPISODE_LOG" >> "$LOG_FILE"
    echo -e "\n--- End of Episode $episode_num ---\n" >> "$LOG_FILE"

    # 5. 실행 결과 추출 및 문장 파싱
    target_color=$(grep -o "target_color='[^']*'" "$EPISODE_LOG" | head -1 | cut -d"'" -f2 || echo "unknown")
    full_instruction=$(grep -o "instruction='[^']*'" "$EPISODE_LOG" | head -1 | cut -d"'" -f2 || echo "unknown")

    # [Verb 파싱]
    verb="unknown"
    for v in "pick up" "grasp" "grab" "catch" "take" "push" "slide" "move"; do
        if [[ "$full_instruction" == *"$v"* ]]; then verb="$v"; break; fi
    done

    # [Noun 파싱]
    noun="unknown"
    for n in "target cylinder" "target box" "cylinder" "box" "cube" "block" "object" "item"; do
        if [[ "$full_instruction" == *"$n"* ]]; then noun="$n"; break; fi
    done

    # [Goal 파싱]
    goal="N/A"
    if [[ "$full_instruction" == *"to the"* || "$full_instruction" == *"toward the"* ]]; then
        for g in "target spot" "destination" "marked point" "final position" "goal"; do
            if [[ "$full_instruction" == *"$g"* ]]; then goal="$g"; break; fi
        done
    fi

    # 6. 성공 여부 판별
    status="✗ FAIL"
    result="실패"
    is_success=0

    if [ $cmd_status -eq 0 ]; then
        if grep -q "lift_success" "$EPISODE_LOG"; then
            status="✓ LIFT_SUCCESS"
            result="성공"
            is_success=1
        else
            ok_count=$(tail -100 "$EPISODE_LOG" | grep -c "\[OK\]" || true)
            if [ "$ok_count" -ge 5 ]; then
                status="⚠ PARTIAL"
                result="성공"
                is_success=1
            else
                status="✗ NO_ACTION"
                result="실패"
            fi
        fi
    else
        if grep -q "IK fail" "$EPISODE_LOG"; then status="✗ IK_FAIL"; else status="✗ CRASH"; fi
        result="실패"
    fi
    
    # 7. CSV 파일에 데이터 쓰기 (성공/실패 여부 포함)
    echo "$episode_num,$target_color,\"$full_instruction\",$verb,$noun,$goal,$result,$status,$elapsed" >> "$CSV_FILE"

    # 8. 실시간 통계 카운트 누적
    # [Color]
    CALL_COUNT["color_$target_color"]=$(( ${CALL_COUNT["color_$target_color"]:-0} + 1 ))
    if [ $is_success -eq 1 ]; then SUCCESS_COUNT["color_$target_color"]=$(( ${SUCCESS_COUNT["color_$target_color"]:-0} + 1 )); fi

    # [Verb]
    CALL_COUNT["verb_$verb"]=$(( ${CALL_COUNT["verb_$verb"]:-0} + 1 ))
    if [ $is_success -eq 1 ]; then SUCCESS_COUNT["verb_$verb"]=$(( ${SUCCESS_COUNT["verb_$verb"]:-0} + 1 )); fi

    # [Noun]
    CALL_COUNT["noun_$noun"]=$(( ${CALL_COUNT["noun_$noun"]:-0} + 1 ))
    if [ $is_success -eq 1 ]; then SUCCESS_COUNT["noun_$noun"]=$(( ${SUCCESS_COUNT["noun_$noun"]:-0} + 1 )); fi

    # [Goal]
    if [ "$goal" != "N/A" ]; then
        CALL_COUNT["goal_$goal"]=$(( ${CALL_COUNT["goal_$goal"]:-0} + 1 ))
        if [ $is_success -eq 1 ]; then SUCCESS_COUNT["goal_$goal"]=$(( ${SUCCESS_COUNT["goal_$goal"]:-0} + 1 )); fi
    fi

    # 전체 카운트
    CALL_COUNT["total"]=$(( ${CALL_COUNT["total"]:-0} + 1 ))
    if [ $is_success -eq 1 ]; then SUCCESS_COUNT["total"]=$(( ${SUCCESS_COUNT["total"]:-0} + 1 )); fi

    # 터미널 출력
    total_ep=${CALL_COUNT["total"]}
    succ_ep=${SUCCESS_COUNT["total"]:-0}
    current_rate=$(( succ_ep * 100 / total_ep ))
    echo "[$status] Color: $target_color | Verb: $verb | Noun: $noun | Goal: $goal"
    echo "          Result: $result | 누적성공률: $current_rate% ($succ_ep/$total_ep)"
    echo ""
done

# 9. 최종 조건별 리포트 요약 함수
print_stat_section() {
    local prefix=$1
    local title=$2
    echo "------------------------------------------"
    echo " $title 통계 리포트 (호출 횟수 및 성공률)"
    echo "------------------------------------------"
    printf "%-18s | %-10s | %-10s\n" "조건 이름" "총 호출수" "성공률"
    echo "------------------------------------------"
    
    for key in "${!CALL_COUNT[@]}"; do
        if [[ "$key" == "$prefix"* ]]; then
            local name=${key#${prefix}_}
            local calls=${CALL_COUNT[$key]}
            local succ=${SUCCESS_COUNT[$key]:-0}
            local rate=$(( succ * 100 / calls ))
            printf "%-18s | %-10d | %-d%%\n" "$name" "$calls" "$rate"
        fi
    done
}

# 10. 터미널 및 통합 로그 파일에 최종 요약 출력
{
    echo ""
    echo "=========================================="
    echo "        배치 롤아웃 조건별 최종 통계"
    echo "=========================================="
    echo "총 에피소드 수 : ${CALL_COUNT["total"]}회"
    echo "총 성공 횟수   : ${SUCCESS_COUNT["total"]:-0}회"
    
    total_calls=${CALL_COUNT["total"]}
    total_succ=${SUCCESS_COUNT["total"]:-0}
    if [ $total_calls -gt 0 ]; then
        final_rate=$(( total_succ * 100 / total_calls ))
    else
        final_rate=0
    fi
    echo "최종 전체 성공률: $final_rate%"
    echo ""

    print_stat_section "color" "TARGET COLOR"
    echo ""
    print_stat_section "verb" "COMMAND VERB"
    echo ""
    print_stat_section "noun" "OBJECT NOUN"
    echo ""
    print_stat_section "goal" "PUSH GOAL"
    echo "=========================================="
    echo "결과 CSV 저장 완료: $CSV_FILE"
    echo "=========================================="
} | tee -a "$LOG_FILE"