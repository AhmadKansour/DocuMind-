
CREATE TABLE public.daily_usage (
    visitor_id TEXT NOT NULL,
    usage_date DATE NOT NULL,
    question_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (visitor_id, usage_date)
);

CREATE TABLE public.global_daily_usage (
    usage_date DATE PRIMARY KEY,
    request_count INTEGER NOT NULL DEFAULT 0
);

CREATE OR REPLACE FUNCTION public.use_ai_request(
    p_visitor_id TEXT,
    p_visitor_limit INTEGER DEFAULT 10,
    p_global_limit INTEGER DEFAULT 100
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
DECLARE
    v_today DATE := (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date;
    v_visitor_count INTEGER;
    v_global_count INTEGER;
BEGIN
    IF p_visitor_id IS NULL
        OR length(p_visitor_id) = 0
        OR length(p_visitor_id) > 128
        OR p_visitor_limit IS NULL
        OR p_global_limit IS NULL
        OR p_visitor_limit < 1
        OR p_global_limit < 1
        OR p_visitor_limit > 10
        OR p_global_limit > 100
    THEN
        RAISE EXCEPTION 'Invalid AI request parameters';
    END IF;

    INSERT INTO public.global_daily_usage (
        usage_date, request_count
    )
    VALUES (v_today, 0)
    ON CONFLICT (usage_date) DO NOTHING;

    SELECT request_count
    INTO v_global_count
    FROM public.global_daily_usage
    WHERE usage_date = v_today
    FOR UPDATE;

    SELECT question_count
    INTO v_visitor_count
    FROM public.daily_usage
    WHERE visitor_id = p_visitor_id
        AND usage_date = v_today;

    v_visitor_count := COALESCE(v_visitor_count, 0);

    IF v_visitor_count >= p_visitor_limit THEN
        RETURN 'visitor_limit';
    END IF;

    IF v_global_count >= p_global_limit THEN
        RETURN 'global_limit';
    END IF;

    INSERT INTO public.daily_usage (
        visitor_id, usage_date, question_count
    )
    VALUES (p_visitor_id, v_today, 1)
    ON CONFLICT (visitor_id, usage_date)
    DO UPDATE SET
        question_count = public.daily_usage.question_count + 1;

    UPDATE public.global_daily_usage
    SET request_count = request_count + 1
    WHERE usage_date = v_today;

    RETURN 'allowed';
END;
$function$;
