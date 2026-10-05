const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const revealTargets = [...document.querySelectorAll("[data-reveal]")];
const lessonVideoData = {
  "lesson-1.html": { lessonLabel: "Lesson 01", startLabel: "0:23:30", startSeconds: 1410 },
  "lesson-2.html": { lessonLabel: "Lesson 02", startLabel: "1:32:49", startSeconds: 5569 },
  "lesson-3.html": { lessonLabel: "Lesson 03", startLabel: "2:15:14", startSeconds: 8114 },
  "lesson-4.html": { lessonLabel: "Lesson 04", startLabel: "4:10:06", startSeconds: 15006 },
  "lesson-5.html": { lessonLabel: "Lesson 05", startLabel: "5:06:01", startSeconds: 18361 },
  "lesson-6.html": { lessonLabel: "Lesson 06", startLabel: "5:53:41", startSeconds: 21221 },
  "lesson-7.html": { lessonLabel: "Lesson 07", startLabel: "7:44:03", startSeconds: 27843 },
  "lesson-8.html": { lessonLabel: "Lesson 08", startLabel: "8:48:46", startSeconds: 31726 },
  "lesson-9.html": { lessonLabel: "Lesson 09", startLabel: "21:47:09", startSeconds: 78429 },
  "lesson-10.html": { lessonLabel: "Lesson 10", startLabel: "22:49:29", startSeconds: 82169 },
};

function initMobileNav() {
  const header = document.querySelector(".site-header");

  if (!header) {
    return;
  }

  const toggle = header.querySelector(".nav-toggle");
  const nav = header.querySelector(".site-nav");

  if (!toggle || !nav) {
    return;
  }

  const mobileNav = window.matchMedia("(max-width: 960px)");
  const navLinks = [...nav.querySelectorAll("a")];

  function isOpen() {
    return document.body.classList.contains("is-nav-open");
  }

  function setOpen(nextOpen) {
    document.body.classList.toggle("is-nav-open", nextOpen);
    toggle.setAttribute("aria-expanded", String(nextOpen));
    toggle.setAttribute("aria-label", nextOpen ? "Close navigation" : "Open navigation");
  }

  function closeNav() {
    setOpen(false);
  }

  toggle.addEventListener("click", () => {
    setOpen(!isOpen());
  });

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      if (mobileNav.matches) {
        closeNav();
      }
    });
  });

  document.addEventListener("click", (event) => {
    if (!mobileNav.matches || !isOpen()) {
      return;
    }

    if (!header.contains(event.target)) {
      closeNav();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isOpen()) {
      closeNav();
    }
  });

  const handleBreakpointChange = (event) => {
    if (!event.matches) {
      closeNav();
    }
  };

  if (typeof mobileNav.addEventListener === "function") {
    mobileNav.addEventListener("change", handleBreakpointChange);
    return;
  }

  mobileNav.addListener(handleBreakpointChange);
}

function initLessonVideoEmbeds() {
  const currentFile = window.location.pathname.split("/").filter(Boolean).pop();
  const videoMeta = currentFile ? lessonVideoData[currentFile] : null;
  const lessonBody = document.querySelector(".lesson-body");

  if (!videoMeta || !lessonBody || document.querySelector(".lesson-video-band")) {
    return;
  }

  const section = document.createElement("section");
  const watchUrl = `https://www.youtube.com/watch?v=SqQUQHdYWyc&t=${videoMeta.startSeconds}s`;
  const embedUrl = `https://www.youtube-nocookie.com/embed/SqQUQHdYWyc?start=${videoMeta.startSeconds}&rel=0`;

  section.className = "lesson-video-band";
  section.setAttribute("aria-labelledby", "lesson-video-title");
  section.innerHTML = `
    <div class="section-shell">
      <div class="lesson-video-card">
        <div class="lesson-video-copy">
          <p class="section-kicker">Lesson Lecture</p>
          <h2 id="lesson-video-title">Start the full course at the beginning of this lesson.</h2>
          <p>
            This player jumps straight to the point where this lesson begins in the full recording, so the lecture and
            deck stay aligned without scrubbing through the entire course.
          </p>
          <div class="lesson-video-meta" aria-label="Lesson video metadata">
            <span>${videoMeta.lessonLabel}</span>
            <span>Starts at ${videoMeta.startLabel}</span>
          </div>
          <a class="hero-video-link lesson-video-link" href="${watchUrl}" target="_blank" rel="noreferrer">
            Open on YouTube
          </a>
        </div>
        <div class="lesson-video-frame hero-video-shell">
          <div class="hero-video-window">
            <iframe
              src="${embedUrl}"
              title="CUDA Programming for NVIDIA H100s lesson segment"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerpolicy="strict-origin-when-cross-origin"
              loading="lazy"
              allowfullscreen
            ></iframe>
          </div>
        </div>
      </div>
    </div>
  `;

  lessonBody.before(section);
}

const pipelineData = [
  {
    id: "descriptor",
    index: "01",
    label: "Descriptors",
    note: "How movement gets described before math happens.",
    accent: "#79d3ff",
    filePath: "files/fast.cu/examples/matmul/matmul_12.cuh",
    fileHref: "files/fast.cu/examples/matmul/matmul_12.cuh",
    lineRange: "Lines 4-49",
    title: "Descriptors make memory movement explicit.",
    description:
      "The course spends real time here because Hopper kernels are shaped long before a tensor core instruction fires. Shared-memory descriptors and tensor maps decide how data lands, how it is swizzled, and what later stages are allowed to assume.",
    points: [
      "Shared-memory addresses get packed into matrix descriptors instead of being treated like ordinary pointers.",
      "The tensor map encodes shape, stride, swizzle, and promotion choices up front so async movement has a precise contract.",
      "This is where the course stops sounding like generic CUDA and starts sounding like Hopper.",
    ],
    stages: [
      { label: "Encode", text: "Pack the shared-memory pointer into descriptor bits." },
      { label: "Shape", text: "Specify global and shared-memory tile geometry explicitly." },
      { label: "Swizzle", text: "Choose the 128B swizzle the downstream pipeline expects." },
      { label: "Dispatch", text: "Hand the descriptor to TMA-backed movement." },
    ],
    snippet: `__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__host__ static inline CUtensorMap create_tensor_map(...) {
    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        gmem_address,
        gmem_prob_shape,
        gmem_prob_stride,
        smem_box_shape,
        smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}`,
  },
  {
    id: "warpgroup",
    index: "02",
    label: "WGMMA",
    note: "Fence, commit, wait, and then issue real tensor-core work.",
    accent: "#a6ff6b",
    filePath: "files/fast.cu/examples/matmul/matmul_12.cuh",
    fileHref: "files/fast.cu/examples/matmul/matmul_12.cuh",
    lineRange: "Lines 20-30, 57-63",
    title: "Warpgroup math is issued like a pipeline, not a single call.",
    description:
      "This is the moment the homepage needs to show, not just name. The file exposes the lifecycle directly: fence the warpgroup, commit the batch, wait on the group, and then feed descriptors into the async WGMMA instruction.",
    points: [
      "The course treats issue, commit, and wait as first-class control flow, not trivia around the instruction.",
      "The shared-memory descriptors built earlier become the operands that make tensor-core issue possible.",
      "Seeing these PTX strings in place is what makes Hopper feel mechanically real.",
    ],
    stages: [
      { label: "Fence", text: "Close out prior shared-memory hazards before issuing math." },
      { label: "Commit", text: "Submit the current warpgroup batch into the async queue." },
      { label: "Wait", text: "Rejoin the dependency stream only when the group is ready." },
      { label: "Issue", text: "Launch WGMMA with descriptor-backed operands." },
    ],
    snippet: `__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}

__device__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");
}

__device__ __forceinline__ void wgmma256(float d[16][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 ..."
    );
}`,
  },
  {
    id: "barriers",
    index: "03",
    label: "Barriers",
    note: "Ordered sequence barriers and shared storage layout.",
    accent: "#ff9b61",
    filePath: "files/sm90_gemm_tma_warpspecialized_pingpong.hpp",
    fileHref: "files/sm90_gemm_tma_warpspecialized_pingpong.hpp",
    lineRange: "Lines 127-185",
    title: "The pipeline is literally laid out in shared storage.",
    description:
      "This file is a strong answer to the question, what do these Hopper pipelines look like in code reality? The answer is: load warp groups, math warp groups, ordered barriers, and storage blocks laid out by name.",
    points: [
      "Warp specialization is not abstract here; the file assigns explicit thread counts to scheduler, load, and MMA roles.",
      "Barriers are typed and staged structures, not vague synchronization advice.",
      "Mainloop, epilogue, and ordering state live together in a shared-memory layout the kernel can reason about directly.",
    ],
    stages: [
      { label: "Split Roles", text: "Separate scheduler, load, epilogue-load, and math responsibilities." },
      { label: "Order", text: "Use ordered sequence barriers to serialize the right handoffs." },
      { label: "Stage", text: "Lay pipeline storage into shared memory with explicit alignment." },
      { label: "Sustain", text: "Keep the ping-pong kernel fed without losing correctness." },
    ],
    snippet: `static constexpr uint32_t NumLoadWarpGroups = 1;
static constexpr uint32_t NumMmaWarpGroups = 2;
static constexpr uint32_t NumMMAThreads = size(TiledMma{});

using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;
using MathWarpGroupOrderBarrier =
  cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup, NumMmaWarpGroups>;

struct SharedStorage {
  struct PipelineStorage : cute::aligned_struct<16, _1> {
    alignas(16) MainloopPipelineStorage mainloop;
    alignas(16) EpiLoadPipelineStorage epi_load;
    alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
    alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
  } pipelines;

  alignas(16) TileSchedulerStorage scheduler;
};`,
  },
  {
    id: "scheduler",
    index: "04",
    label: "Stream-K",
    note: "Work tiles, splits, reduction units, and persistent stepping.",
    accent: "#7fffd4",
    filePath: "files/sm90_tile_scheduler_stream_k.hpp",
    fileHref: "files/sm90_tile_scheduler_stream_k.hpp",
    lineRange: "Lines 44-153",
    title: "Scheduling is part of the algorithm, not the scaffolding around it.",
    description:
      "The scheduler file makes that explicit. Work tiles have identity, split state, reduction behavior, and final-split logic. The course keeps emphasizing this because utilization is often a scheduling story as much as a math story.",
    points: [
      "Stream-K decomposition gives each unit of work a precise slice of K and a notion of what remains.",
      "Separate reduction units are represented directly in the work-tile structure, not hand-waved away.",
      "Persistent kernels need bookkeeping that mirrors the machine's notion of ongoing work.",
    ],
    stages: [
      { label: "Define", text: "Model the output tile and its K-slice responsibilities." },
      { label: "Split", text: "Track how much of the tile is assigned to this work unit." },
      { label: "Reduce", text: "Handle separate-reduction units explicitly when needed." },
      { label: "Finish", text: "Know when the unit is the final split for that output tile." },
    ],
    snippet: `// Persistent Thread Block scheduler leveraging stream-K decomposition
class PersistentTileSchedulerSm90StreamK {
  dim3 block_id_in_cluster_;
  uint64_t current_work_linear_idx_ = 0;
  uint32_t unit_iter_start_ = 0;

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t K_idx = 0;
    uint32_t k_tile_count = 0;
    uint32_t k_tile_remaining = 0;
    bool is_separate_reduction = false;

    bool is_valid() const {
      return k_tile_count > 0 || is_separate_reduction;
    }

    bool is_final_split(uint32_t k_tiles_per_output_tile) const {
      return (K_idx + k_tile_count) == k_tiles_per_output_tile;
    }
  };
};`,
  },
];

revealTargets.forEach((element, index) => {
  element.style.setProperty("--reveal-order", String(index % 4));
});

function initRevealAnimations() {
  if (!prefersReducedMotion.matches) {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }

          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -8% 0px",
      },
    );

    revealTargets.forEach((element) => {
      revealObserver.observe(element);
    });
    return;
  }

  revealTargets.forEach((element) => {
    element.classList.add("is-visible");
  });
}

function initPipelineShowcase() {
  const root = document.querySelector("[data-pipeline-root]");

  if (!root) {
    return;
  }

  const selector = root.querySelector("[data-pipeline-selector]");
  const stages = root.querySelector("[data-pipeline-stages]");
  const pathElement = root.querySelector("[data-pipeline-path]");
  const titleElement = root.querySelector("[data-pipeline-title]");
  const descriptionElement = root.querySelector("[data-pipeline-description]");
  const pointsElement = root.querySelector("[data-pipeline-points]");
  const linkElement = root.querySelector("[data-pipeline-link]");
  const linesElement = root.querySelector("[data-pipeline-lines]");
  const fileTagElement = root.querySelector("[data-pipeline-filetag]");
  const lineTagElement = root.querySelector("[data-pipeline-linetag]");
  const codeElement = root.querySelector("[data-pipeline-code]");
  const codeWindow = root.querySelector(".pipeline-code-window");
  const panel = root.querySelector(".pipeline-panel");

  let activeIndex = 0;
  let autoRotateId = null;
  let hasUserInteracted = false;
  let isInView = false;

  const buttons = pipelineData.map((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pipeline-button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", "false");
    button.innerHTML = `
      <span class="pipeline-button-index">${item.index}</span>
      <span class="pipeline-button-label">${item.label}</span>
      <span class="pipeline-button-note">${item.note}</span>
    `;
    button.addEventListener("click", () => {
      hasUserInteracted = true;
      stopAutoRotate();
      renderConcept(index);
    });
    selector.append(button);
    return button;
  });

  function renderStages(items) {
    stages.innerHTML = items
      .map(
        (stage, index) => `
          <article class="pipeline-stage" style="transition-delay:${index * 70}ms">
            <span class="pipeline-stage-index">${String(index + 1).padStart(2, "0")}</span>
            <h4>${stage.label}</h4>
            <p>${stage.text}</p>
          </article>
        `,
      )
      .join("");
  }

  function renderPoints(items) {
    pointsElement.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
  }

  function animateUpdate() {
    if (prefersReducedMotion.matches) {
      return;
    }

    const animationConfig = {
      duration: 360,
      easing: "cubic-bezier(0.18, 0.9, 0.22, 1)",
    };

    panel.animate(
      [
        { opacity: 0.6, transform: "translateY(12px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      animationConfig,
    );

    codeWindow.animate(
      [
        { opacity: 0.5, transform: "translateY(12px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      animationConfig,
    );

    stages.animate(
      [
        { opacity: 0.35, transform: "translateY(10px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      animationConfig,
    );
  }

  function renderConcept(index) {
    const item = pipelineData[index];
    activeIndex = index;

    root.style.setProperty("--pipeline-accent", item.accent);

    buttons.forEach((button, buttonIndex) => {
      const isActive = buttonIndex === index;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });

    pathElement.textContent = item.filePath;
    titleElement.textContent = item.title;
    descriptionElement.textContent = item.description;
    renderPoints(item.points);
    renderStages(item.stages);
    linkElement.href = item.fileHref;
    linesElement.textContent = item.lineRange;
    fileTagElement.textContent = item.filePath;
    lineTagElement.textContent = item.lineRange;
    codeElement.textContent = item.snippet;

    animateUpdate();
  }

  function stopAutoRotate() {
    if (autoRotateId) {
      window.clearInterval(autoRotateId);
      autoRotateId = null;
    }
  }

  function startAutoRotate() {
    if (prefersReducedMotion.matches || hasUserInteracted || !isInView) {
      return;
    }

    stopAutoRotate();
    autoRotateId = window.setInterval(() => {
      renderConcept((activeIndex + 1) % pipelineData.length);
    }, 6500);
  }

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        isInView = entry.isIntersecting;

        if (isInView) {
          startAutoRotate();
        } else {
          stopAutoRotate();
        }
      });
    },
    { threshold: 0.35 },
  );

  sectionObserver.observe(root);

  root.addEventListener("mouseenter", stopAutoRotate);
  root.addEventListener("mouseleave", startAutoRotate);
  root.addEventListener("focusin", stopAutoRotate);
  root.addEventListener("focusout", startAutoRotate);

  renderConcept(0);
}

function initFaqAccordions() {
  const items = [...document.querySelectorAll("[data-faq-item]")];

  if (items.length === 0) {
    return;
  }

  function setOpen(item, shouldOpen) {
    const trigger = item.querySelector(".faq-trigger");
    const panel = item.querySelector(".faq-panel");

    if (!trigger || !panel) {
      return;
    }

    item.classList.toggle("is-open", shouldOpen);
    trigger.setAttribute("aria-expanded", String(shouldOpen));
    panel.hidden = !shouldOpen;
  }

  items.forEach((item, index) => {
    setOpen(item, index === 0);

    const trigger = item.querySelector(".faq-trigger");

    if (!trigger) {
      return;
    }

    trigger.addEventListener("click", () => {
      const shouldOpen = trigger.getAttribute("aria-expanded") !== "true";

      items.forEach((otherItem) => {
        setOpen(otherItem, false);
      });

      if (shouldOpen) {
        setOpen(item, true);
      }
    });
  });
}

function initSidebarScrollSpy() {
  const sideNav = document.querySelector(".lesson-side-nav");

  if (!sideNav) {
    return;
  }

  const navLinks = [...sideNav.querySelectorAll("a[href^='#']")];

  if (navLinks.length === 0) {
    return;
  }

  const sections = navLinks
    .map((link) => {
      const id = link.getAttribute("href").slice(1);
      const section = document.getElementById(id);
      return section ? { link, section } : null;
    })
    .filter(Boolean);

  if (sections.length === 0) {
    return;
  }

  let currentActive = null;

  const observer = new IntersectionObserver(
    (entries) => {
      let topMost = null;
      let topMostY = Infinity;

      sections.forEach(({ section }) => {
        const rect = section.getBoundingClientRect();

        if (rect.top < window.innerHeight * 0.4 && rect.bottom > 0) {
          if (rect.top < topMostY) {
            topMostY = rect.top;
            topMost = section;
          }
        }
      });

      if (!topMost) {
        return;
      }

      const match = sections.find((s) => s.section === topMost);

      if (match && match.link !== currentActive) {
        if (currentActive) {
          currentActive.classList.remove("is-active");
        }

        match.link.classList.add("is-active");
        currentActive = match.link;
      }
    },
    { threshold: 0, rootMargin: "-40% 0px -40% 0px" },
  );

  sections.forEach(({ section }) => observer.observe(section));

  // Also update on scroll for more responsive feel
  let scrollTicking = false;

  window.addEventListener("scroll", () => {
    if (scrollTicking) {
      return;
    }

    scrollTicking = true;
    requestAnimationFrame(() => {
      let topMost = null;
      let topMostY = Infinity;

      sections.forEach(({ link, section }) => {
        const rect = section.getBoundingClientRect();

        if (rect.top <= window.innerHeight * 0.4 && rect.bottom > 0) {
          if (rect.top < topMostY || (rect.top <= 0 && Math.abs(rect.top) < Math.abs(topMostY))) {
            topMostY = rect.top;
            topMost = link;
          }
        }
      });

      if (topMost && topMost !== currentActive) {
        if (currentActive) {
          currentActive.classList.remove("is-active");
        }

        topMost.classList.add("is-active");
        currentActive = topMost;
      }

      scrollTicking = false;
    });
  }, { passive: true });
}

function initLanguageSelect() {
  const selects = document.querySelectorAll("[data-lang-select] select");

  for (const select of selects) {
    select.addEventListener("change", () => {
      if (select.value) {
        window.location.assign(select.value);
      }
    });
  }
}

initMobileNav();
initLessonVideoEmbeds();
initRevealAnimations();
initPipelineShowcase();
initFaqAccordions();
initSidebarScrollSpy();
initLanguageSelect();
