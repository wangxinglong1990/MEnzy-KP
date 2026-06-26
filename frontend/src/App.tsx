import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import { AppShell } from "./components/layout/AppShell";
import { HomePage } from "./pages/HomePage";
import { PredictionPage } from "./pages/PredictionPage";
import { BatchPage } from "./pages/BatchPage";
import { RankingPage } from "./pages/RankingPage";
import { ClusteringPage } from "./pages/ClusteringPage";
import { DockingPage } from "./pages/DockingPage";
import { TrainingPage } from "./pages/TrainingPage";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 10000 } },
});

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <HomePage /> },
      { path: "/predict", element: <PredictionPage /> },
      { path: "/batch", element: <BatchPage /> },
      { path: "/ranking", element: <RankingPage /> },
      { path: "/clustering", element: <ClusteringPage /> },
      { path: "/docking", element: <DockingPage /> },
      { path: "/training", element: <TrainingPage /> },
    ],
  },
]);

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster position="bottom-right" />
    </QueryClientProvider>
  );
}
