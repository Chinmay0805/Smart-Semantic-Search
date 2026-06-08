"use client";

import { useEffect, useState } from "react";
import { Check } from "lucide-react";

const PIPELINE_STEPS = [
  "Embedding query",
  "Routing to GMM Cluster",
  "Checking Cache",
  "Retrieving vectors",
] as const;

interface PipelineProgressProps {
  isActive: boolean;
}

export default function PipelineProgress({ isActive }: PipelineProgressProps) {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setActiveStep(0);
      return;
    }

    setActiveStep(0);
    const interval = window.setInterval(() => {
      setActiveStep((prev) =>
        prev < PIPELINE_STEPS.length - 1 ? prev + 1 : prev
      );
    }, 450);

    return () => window.clearInterval(interval);
  }, [isActive]);

  if (!isActive) return null;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 px-5 py-4">
      <ol className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-0">
        {PIPELINE_STEPS.map((step, index) => {
          const isComplete = index < activeStep;
          const isCurrent = index === activeStep;

          return (
            <li key={step} className="flex items-center sm:flex-1">
              <div className="flex items-center gap-2.5">
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-medium ${
                    isComplete
                      ? "bg-blue-600 text-white"
                      : isCurrent
                        ? "border border-blue-500 text-blue-500"
                        : "border border-slate-700 text-slate-600"
                  }`}
                >
                  {isComplete ? <Check className="h-3 w-3" /> : index + 1}
                </span>
                <span
                  className={`text-sm ${
                    isCurrent
                      ? "font-medium text-slate-200"
                      : isComplete
                        ? "text-slate-400"
                        : "text-slate-600"
                  }`}
                >
                  {step}
                  {isCurrent && "…"}
                </span>
              </div>
              {index < PIPELINE_STEPS.length - 1 && (
                <span
                  className="mx-3 hidden h-px flex-1 bg-slate-800 sm:block"
                  aria-hidden
                />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
