-- ShipWise AI: user profiles on Supabase. Run once in the Supabase SQL Editor.
--
-- Supabase Auth already stores the email and the password hash (in auth.users), so
-- there is no password table here. This adds one table, `profiles`, holding what the
-- rest of ShipWise needs to know about a signed-in PM: their name and their company.
-- `company_id` is what the recommendation store and the learning stage group reports
-- by ("calibrated to your company's data"), so it is derived from the company name
-- typed at sign-up: lower-case, letters and digits only, words joined by hyphens.
-- (Demo shortcut: two people who spell the same company differently end up as two
-- companies. A real build would have an organisations table and invitations.)

create table if not exists public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  full_name   text not null,
  company     text not null,
  company_id  text not null,
  created_at  timestamptz not null default now()
);

-- Row-level security: a signed-in user can read and edit only their own row. The web
-- page uses the public (anon) key, so this is what stops one PM reading another's profile.
alter table public.profiles enable row level security;

drop policy if exists "read own profile" on public.profiles;
create policy "read own profile" on public.profiles
  for select using (auth.uid() = id);

-- No insert and no update policy, on purpose. The row is created by the trigger below and
-- never by the browser, and nothing in the app edits a profile. An update policy would let
-- a user change their own company_id to another company's, which would put them in that
-- company's group for the learning stage (found in testing on 2026-10-04, when an earlier
-- version of this file had one). The line below removes that policy if it exists.
drop policy if exists "update own profile" on public.profiles;

-- When someone signs up, create their profile from the name and company they entered.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
declare
  company_text text := coalesce(nullif(trim(new.raw_user_meta_data ->> 'company'), ''), 'Unknown company');
begin
  insert into public.profiles (id, full_name, company, company_id)
  values (
    new.id,
    coalesce(nullif(trim(new.raw_user_meta_data ->> 'full_name'), ''), split_part(new.email, '@', 1)),
    company_text,
    trim(both '-' from regexp_replace(lower(company_text), '[^a-z0-9]+', '-', 'g'))
  );
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
