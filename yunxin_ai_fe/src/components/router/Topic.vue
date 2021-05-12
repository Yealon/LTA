<template>
  <div>
    <s-header title="接入云心AI服务"></s-header>
    <pull-down ref="pulldown"></pull-down>

    <div class="article-wrap" style="margin: 60px 10px 0">
      <h3 class="title">获取鉴权</h3>
      <div class="repo-env">
        <div class="row">
          <div class="col-xs-3 text-right">URL：</div>
          <div class="col-xs-8">ai.yunxin.plus/oauth/[username]/[password]</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">请求方法：</div>
          <div class="col-xs-8">POST</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">传入参数：</div>
          <div class="col-xs-8">username: 用户名</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right"></div>
          <div class="col-xs-8">password: 密码</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">输出参数：</div>
          <div class="col-xs-8">code：错误码。20000：正确。</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right"></div>
          <div class="col-xs-8">data：token</div>
        </div>
      </div>

      <h3 class="title">AI能力请求</h3>
      <div class="repo-env">
        <div class="row">
          <div class="col-xs-3 text-right">URL：</div>
          <div class="col-xs-8">ai.yunxin.plus/ap1/v1/[AI能力key]</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">请求方法：</div>
          <div class="col-xs-8">GET</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">请求头：</div>
          <div class="col-xs-8">token: Brearer token</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">传入参数：</div>
          <div class="col-xs-8">text: [Query params]案件描述</div>
        </div>
        <div class="row">
          <div class="col-xs-3 text-right">输出参数：</div>
          <div class="col-xs-8">data：[标签1, 标签2, 标签3]</div>
        </div>
      </div>
    </div>
    <router-view></router-view>
  </div>
</template>

<script>
import { getToken, TOKEN_CACHE_KEY } from "../TokenUtils";
import { URL_TOPIC_LIST } from "../Config";
import axios from "axios";
import PullDown from "../PullDown.vue";
import { bindPullUpLoading, unbindPullUpLoading } from "../PullUpLoading";

export default {
  name: "topic",
  data() {
    return {
      // 分页使用
      hasMore: false,
      p: 1,

      loading: false, // 是否正在加载

      topics: [],
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
      callback && callback();
      return;
      if (this.$route.path != "/topics") {
        return;
      }

      if (append && !this.hasMore) {
        return;
      }

      this.loading = true;

      if (!append) {
        this.topics = [];
        this.p = 1;
      }

      getToken().then((token) => {
        axios
          .get(URL_TOPIC_LIST, {
            params: {
              token: token,
              p: this.p,
            },
          })
          .then((resp) => {
            if (resp.abort) return;

            var data = resp.data.data;
            var topics = data.topics;

            this.hasMore = data.has_more;

            if (append) {
              topics.forEach((topic) => {
                this.topics.push(topic);
              });
            } else {
              this.topics = topics;
            }

            this.p++;
          })
          .finally(() => {
            this.loading = false;
            typeof callback === "function" && callback();
          });
      });
    },
    openTopicDetail: function (topic) {
      this.$router.push({
        name: "topic-detail",
        params: { id: topic.tid },
      });
    },
    formatTime: function (val) {
      let date = new Date(val);
      return date.getFullYear() + "-" + date.getMonth() + "-" + date.getDate();
    },
  },
};
</script>

<style lang="sass" scoped>
@import "List.scss"
</style>