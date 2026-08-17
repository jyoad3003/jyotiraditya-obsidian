---
tags:
  - weekly-review
  - periodic-review
  - type/review
created: {{date}}
week: {{title}} # e.g. 2026-W34
rating: ⭐️⭐️⭐️⭐️⭐️
---

# 🗓️ Weekly Review: {{title}}

> [!quote] "Reviewing the past week is the compass that guides the next."

---

## 🏆 Highlights & Big Wins
- 
- 
- 

---

## 📊 Retrospective: What Worked vs What Didn't
### 🟢 What Went Well?
- 

### 🔴 What Caused Friction or Obstacles?
- 

### 💡 Key Insights & Lessons Learned
- 

---

## 🎯 Active Projects & Goals Status

```dataview
TABLE status AS "Status", deadline AS "Deadline", progress AS "Progress"
FROM "Projects" OR #project
WHERE status != "Completed" AND status != "Archived"
SORT deadline ASC
```

---

## 🚀 Objectives & Focus for Next Week (Top 3)
1. 🎯 **Objective 1**: 
2. 🎯 **Objective 2**: 
3. 🎯 **Objective 3**: 

---

## 📝 Notes Created This Week (Dataview)

```dataview
TABLE file.cday AS "Created Date", file.folder AS "Folder"
WHERE file.cday >= date(today) - dur(7 days)
SORT file.cday DESC
```
