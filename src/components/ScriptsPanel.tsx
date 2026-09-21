import { useState } from "react";
import { Check, Copy, Download, FileCode2 } from "lucide-react";

import { PYTHON_FILES } from "@/lib/python-scripts";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function downloadText(name: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export function ScriptsPanel() {
  const [activeName, setActiveName] = useState(PYTHON_FILES[0]!.name);
  const [copied, setCopied] = useState(false);
  const active = PYTHON_FILES.find((f) => f.name === activeName) ?? PYTHON_FILES[0]!;

  const copy = async () => {
    await navigator.clipboard.writeText(active.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  const downloadAll = () => {
    for (const file of PYTHON_FILES) downloadText(file.name, file.code);
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[16rem_1fr]">
      <div className="space-y-2">
        {PYTHON_FILES.map((file) => (
          <button
            key={file.name}
            onClick={() => setActiveName(file.name)}
            className={cn(
              "w-full rounded-lg border px-3 py-2 text-left transition-colors",
              file.name === active.name
                ? "border-primary bg-primary/15"
                : "border-border bg-card hover:bg-secondary",
            )}
          >
            <div className="flex items-center gap-2 font-mono text-sm text-foreground">
              <FileCode2 className="h-3.5 w-3.5 text-primary" />
              {file.name}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">{file.purpose}</div>
          </button>
        ))}
        <Button variant="secondary" className="w-full" onClick={downloadAll}>
          <Download className="mr-2 h-4 w-4" /> Download all files
        </Button>
      </div>

      <div className="min-w-0 rounded-xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <span className="font-mono text-sm text-foreground">{active.name}</span>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={copy}>
              {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => downloadText(active.name, active.code)}>
              <Download className="mr-1.5 h-3.5 w-3.5" /> Save
            </Button>
          </div>
        </div>
        <pre className="max-h-[32rem] overflow-auto p-4 font-mono text-xs leading-relaxed text-foreground">
          {active.code}
        </pre>
      </div>
    </div>
  );
}
