<template>
  <div>
    <s-header title="云心AI能力展示"></s-header>
    <div id="list-main-wrap">
      <div
        class="list-box list-box_text"
        v-for="article in articles"
        @click="openArticleDetail(article)"
      >
        <h4 class="list-box__title">{{ article.title }}</h4>
        <ul class="list-box__info">
          <li class="list-box__info__meta">{{ article.name }}</li>
          <li class="list-box__info__meta">接入数 {{ article.viewnum }}</li>
          <li class="list-box__info__meta list-box__info__meta_extra">
            <button class="v-modal-btn primary" style="font-size: 13px">
              预览效果
            </button>
          </li>
        </ul>
      </div>
      <div class="load-more" @click="loadData(true)">
        {{
          hasMore ? (loading ? "正在加载更多" : "点击加载更多") : "没有更多了"
        }}
      </div>
    </div>

    <pull-down ref="pulldown"></pull-down>

    <router-view></router-view>
  </div>
</template>

<script>
import Vue from "vue";
import EventBus from "../../components/EventBus";

import { getToken, TOKEN_CACHE_KEY } from "../TokenUtils";
import { URL_ARTICLE_LIST } from "../Config";
import axios from "axios";
import PullDown from "../PullDown.vue";
import { bindPullUpLoading, unbindPullUpLoading } from "../PullUpLoading";

export default {
  name: "article",
  data() {
    return {
      // 分页使用
      hasMore: false,
      base: undefined,

      loading: false, // 是否正在加载

      articles: [],
    };
  },
  components: {
    PullDown,
  },
  mounted: function () {
    this.loadData();
  },
  beforeRouteEnter: function (to, from, next) {
    next((vm) => {
      vm.$refs.pulldown.bindElement(null, () => {
        return new Promise((resolve, reject) => {
          vm.loadData(false, () => {
            resolve();
          });
        });
      });

      bindPullUpLoading(
        () => {
          return new Promise((resolve, reject) => {
            vm.loadData(true, () => {
              resolve();
            });
          });
        },
        { preloadHeight: 300 }
      ); // 100 * 3 三张卡片的高度
    });
  },
  beforeRouteLeave: function (to, from, next) {
    this.$refs.pulldown.unbind();

    unbindPullUpLoading();
    next();
  },
  methods: {
    loadData: function (append, callback) {
      if (this.$route.path != "/articles") {
        return;
      }
      if (append && !this.hasMore) {
        return;
      }

      this.loading = true;

      if (!append) {
        this.articles = [];
        this.base = undefined;
      }

      this.articles = [
        {
          id: 1,
          title: "法律案由预测",
          name: "由案件描述生成关键词",
          viewnum: 324,
        },
        {
          id: -1,
          title: "一键抠图",
          name: "识别主体并生成PNG图片",
          viewnum: 0,
        },
        {
          id: -1,
          title: "图片文字识别",
          name: "一键复制图片中的文字",
          viewnum: 0,
        },
      ];
      this.loading = false;
      return;

      getToken().then((token) => {
        axios
          .get(URL_ARTICLE_LIST, {
            params: {
              token: token,
              base: this.base,
            },
          })
          .then((resp) => {
            if (resp.abort) return;

            var data = resp.data.data;
            var articles = data.articles;

            this.hasMore = data.has_more;

            if (append) {
              articles.forEach((article) => {
                this.articles.push(article);
              });
            } else {
              this.articles = articles;
            }

            let len = this.articles.length;
            if (len > 0) {
              this.base = this.articles[len - 1].id;
            }
          })
          .finally(() => {
            this.loading = false;
            typeof callback === "function" && callback();
          });
      });
    },
    openArticleDetail: function (article) {
      if (article.id === -1) {
        EventBus.$emit("hideLoading"); // 隐藏loading
        EventBus.$emit("alert", { message: "开发中，敬请期待！" });
        return;
      }
      this.$router.push({
        name: "article-detail",
        params: { id: article.id },
      });
    },
  },
};
</script>

<style lang="sass" scoped>
@import "List.scss"
</style>