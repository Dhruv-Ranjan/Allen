create table if not exists public.student_weekly_analysis (
    student_id text not null,
    week smallint not null check (week between 1 and 9),
    analysis jsonb not null,
    primary key (student_id, week)
);

alter table public.student_weekly_analysis enable row level security;

-- Do not add public or anon read policies for student records.