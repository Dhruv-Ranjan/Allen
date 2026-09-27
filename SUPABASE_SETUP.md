# Supabase Student Analysis

The existing pipeline generates synthetic student records and nine weeks of
weekly risk analysis. The integration stores one row per student per week;
each row's `analysis` JSON contains both the original weekly measurements and
all trajectory and signal-level results.

## Setup

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. In the Supabase SQL Editor, run `supabase_schema.sql`.
3. Copy `.env.example` to `.env`, then fill in the project URL and a server-side
   secret key. Keep `.env` private; never put this key in browser code or commit it.
4. Generate the CSV outputs with `python student_wellbeing_ai.py`.
5. Upload or look up a student:

   - `python supabase_student_lookup.py upload`
   - `python supabase_student_lookup.py lookup 7`
   - `python supabase_student_lookup.py` prompts for a student number.

Uploads are safe to repeat: `(student_id, week)` is the primary key and rows
are upserted. Student numbers such as `7` are normalized to `STU-007`.
Looking up a student prints every stored weekly analysis record, ordered by
week.

## Privacy

The supplied dataset is synthetic. The analysis is a behavioral early-warning
signal, not a medical or psychological diagnosis. Before using real student
data, verify school consent, access controls, retention, and applicable privacy
requirements. The table has row-level security enabled and deliberately has no
public read policy; use the secret key only in this trusted server-side script.